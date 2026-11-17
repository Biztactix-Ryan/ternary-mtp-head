// llama-dump-nextn: teacher-forced dump of trunk hidden states (h_nextn, post output_norm)
// and the trunk's top-K next-token candidates, for training an MTP draft head.
//
// input : JSONL, one sequence per line: {"id": ..., "prompt": ..., "completion": ...}
// output: <out>/<prefix>-NNNN.{tok.i32,h.f16,topk.i32,topk.f16,index.jsonl}
//         with --layers a,b,c also aux.f16: per position the inputs of those trunk layers, concatenated
//         (what a DFlash/DSpark drafter reads; pass the draft's extract ids = its target_layer_ids + 1)

#include "llama.h"
#include "../../src/llama-ext.h"
#include "ggml.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <thread>
#include <utility>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool add_special) {
    if (text.empty()) {
        return {};
    }
    int n = -llama_tokenize(vocab, text.c_str(), text.size(), NULL, 0, add_special, true);
    std::vector<llama_token> out(n);
    if (n > 0 && llama_tokenize(vocab, text.c_str(), text.size(), out.data(), out.size(), add_special, true) < 0) {
        fprintf(stderr, "tokenize failed\n");
        exit(1);
    }
    return out;
}

struct shard_writer {
    std::string dir, prefix;
    int         idx = -1;
    int64_t     n   = 0; // positions in the current shard
    int64_t     max_positions;
    FILE *      f_tok = nullptr, * f_h = nullptr, * f_ti = nullptr, * f_tp = nullptr, * f_idx = nullptr, * f_aux = nullptr;
    bool        aux = false;

    FILE * open(const char * ext) {
        char name[64];
        snprintf(name, sizeof(name), "%s-%04d.%s", prefix.c_str(), idx, ext);
        std::string path = dir + "/" + name;
        FILE * f = fopen(path.c_str(), "wb");
        if (!f) {
            fprintf(stderr, "cannot open %s\n", path.c_str());
            exit(1);
        }
        return f;
    }
    void close() {
        for (FILE * f : { f_tok, f_h, f_ti, f_tp, f_idx, f_aux }) {
            if (f) {
                fclose(f);
            }
        }
        f_tok = f_h = f_ti = f_tp = f_idx = f_aux = nullptr;
    }
    // called at a sequence boundary
    void begin_sequence() {
        if (idx < 0 || n >= max_positions) {
            close();
            idx++;
            n     = 0;
            f_tok = open("tok.i32");
            f_h   = open("h.f16");
            f_ti  = open("topk.i32");
            f_tp  = open("topk.f16");
            f_idx = open("index.jsonl");
            f_aux = aux ? open("aux.f16") : nullptr;
        }
    }
};

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string model_path, in_path, out_dir, prefix = "shard";
    int         ngl = 99, n_ctx = 8192, n_batch = 512, top_k = 32, max_prompt_rows = 0;
    int64_t     shard_positions = 100000;
    std::vector<uint32_t> layers;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", a.c_str());
                exit(1);
            }
            return argv[++i];
        };
        if (a == "-m") {
            model_path = next();
        } else if (a == "-i") {
            in_path = next();
        } else if (a == "-o") {
            out_dir = next();
        } else if (a == "--prefix") {
            prefix = next();
        } else if (a == "-ngl") {
            ngl = atoi(next());
        } else if (a == "-c") {
            n_ctx = atoi(next());
        } else if (a == "-b") {
            n_batch = atoi(next());
        } else if (a == "--topk") {
            top_k = atoi(next());
        } else if (a == "--max-prompt-rows") {
            max_prompt_rows = atoi(next());
        } else if (a == "--layers") {
            std::string v = next();
            for (size_t b = 0; b < v.size();) {
                size_t e = v.find(',', b);
                if (e == std::string::npos) {
                    e = v.size();
                }
                layers.push_back((uint32_t) atoi(v.substr(b, e - b).c_str()));
                b = e + 1;
            }
        } else if (a == "--shard-positions") {
            shard_positions = atoll(next());
        } else {
            fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 1;
        }
    }
    if (model_path.empty() || in_path.empty() || out_dir.empty()) {
        fprintf(stderr, "usage: %s -m model.gguf -i in.jsonl -o outdir [--prefix shard] [-c 8192] [-b 512] [--topk 32] [-ngl 99]\n", argv[0]);
        return 1;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers       = ngl;
    llama_model * model        = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "unable to load model\n");
        return 1;
    }
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);
    const int           n_embd  = llama_model_n_embd(model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx                = n_ctx;
    cparams.n_batch              = n_batch;
    cparams.n_ubatch             = n_batch;
    cparams.no_perf              = true;
    llama_context * ctx          = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }
    llama_set_embeddings_nextn(ctx, true, /*masked=*/ false);
    for (uint32_t lid : layers) {
        llama_set_embeddings_layer_inp(ctx, lid, true);
    }

    llama_batch_ext * batch = llama_batch_ext_init(ctx);

    shard_writer w;
    w.dir           = out_dir;
    w.prefix        = prefix;
    w.max_positions = shard_positions;
    w.aux           = !layers.empty();
    std::vector<ggml_fp16_t> baux;

    std::vector<ggml_fp16_t> bh, btp;
    std::vector<int32_t>     bti;
    const int n_threads = std::max(1u, std::min(24u, std::thread::hardware_concurrency() / 4));

    std::ifstream in(in_path);
    std::string   line;
    int64_t       n_seq = 0, n_pos_total = 0;
    const int64_t t0 = ggml_time_us();

    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        json j = json::parse(line);
        const std::string id = j.value("id", std::to_string(n_seq));

        std::vector<llama_token> toks = tokenize(vocab, j.value("prompt", ""), true);
        int32_t                  n_prompt = toks.size();
        {
            // exact generated token ids, when the generator logged them; otherwise re-tokenise the text
            std::vector<llama_token> c;
            if (j.contains("completion_tokens") && j["completion_tokens"].is_array() && !j["completion_tokens"].empty()) {
                c = j["completion_tokens"].get<std::vector<llama_token>>();
            } else {
                c = tokenize(vocab, j.value("completion", ""), false);
            }
            toks.insert(toks.end(), c.begin(), c.end());
        }
        if ((int) toks.size() > n_ctx) {
            const int drop = toks.size() - n_ctx;
            toks.erase(toks.begin(), toks.begin() + drop);
            n_prompt = std::max(0, n_prompt - drop);
        }
        const int n = toks.size();
        if (n < 2) {
            continue;
        }

        // with --max-prompt-rows N only the last N prompt positions and the completion are written;
        // the earlier prompt is still decoded so the hidden states see the full context
        const int keep_from = max_prompt_rows > 0 ? std::max(0, n_prompt - max_prompt_rows) : 0;

        w.begin_sequence();
        llama_memory_clear(llama_get_memory(ctx), true);

        for (int off = 0; off < n; off += n_batch) {
            const int nb = std::min(n_batch, n - off);
            llama_batch_ext_clear(batch);
            for (int i = 0; i < nb; i++) {
                const int32_t   idx = llama_batch_ext_add_token(batch, 0, toks[off + i]);
                const llama_pos pos = off + i;
                llama_batch_ext_set_pos(batch, idx, &pos);
                llama_batch_ext_set_output_logits(batch, idx, true);
            }
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) {
                fprintf(stderr, "llama_process failed (seq %s, offset %d)\n", id.c_str(), off);
                return 1;
            }
            // rows before keep_from are context only: decoded, not written
            const int i0 = std::max(0, keep_from - off);
            if (i0 >= nb) {
                continue;
            }
            const int nw = nb - i0;
            // per-row post-processing (float16 conversion, log-softmax, top-K) on worker threads
            bh.resize((size_t) nw * n_embd);
            bti.resize((size_t) nw * top_k);
            btp.resize((size_t) nw * top_k);
            std::atomic<int>  next(i0);
            std::atomic<bool> bad(false);
            auto work = [&]() {
                std::vector<std::pair<float, int32_t>> heap;
                std::vector<float>                     lpv(top_k);
                const auto better = [](const std::pair<float, int32_t> & a, const std::pair<float, int32_t> & b) {
                    return a.first > b.first || (a.first == b.first && a.second < b.second);
                };
                for (int i; (i = next.fetch_add(1)) < nb;) {
                    const float * h = llama_get_embeddings_nextn_ith(ctx, i);
                    const float * l = llama_get_logits_ith(ctx, i);
                    if (!h || !l) {
                        bad = true;
                        return;
                    }
                    const size_t o = (size_t) (i - i0);
                    ggml_fp32_to_fp16_row(h, bh.data() + o * n_embd, n_embd);
                    float mx = l[0];
                    for (int v = 1; v < n_vocab; v++) {
                        mx = std::max(mx, l[v]);
                    }
                    double sum = 0.0;
                    heap.clear();
                    for (int v = 0; v < n_vocab; v++) {
                        sum += exp((double) (l[v] - mx));
                        const std::pair<float, int32_t> e(l[v], v);
                        if ((int) heap.size() < top_k) {
                            heap.push_back(e);
                            std::push_heap(heap.begin(), heap.end(), better);   // worst candidate on top
                        } else if (better(e, heap.front())) {
                            std::pop_heap(heap.begin(), heap.end(), better);
                            heap.back() = e;
                            std::push_heap(heap.begin(), heap.end(), better);
                        }
                    }
                    const float lse = mx + (float) log(sum);
                    std::sort(heap.begin(), heap.end(), better);
                    for (int k = 0; k < top_k; k++) {
                        bti[o * top_k + k] = heap[k].second;
                        lpv[k]             = heap[k].first - lse;
                    }
                    ggml_fp32_to_fp16_row(lpv.data(), btp.data() + o * top_k, top_k);
                }
            };
            {
                std::vector<std::thread> th;
                for (int t = 0; t < n_threads; t++) {
                    th.emplace_back(work);
                }
                for (auto & t : th) {
                    t.join();
                }
            }
            if (bad) {
                fprintf(stderr, "missing output row (seq %s)\n", id.c_str());
                return 1;
            }
            fwrite(toks.data() + off + i0, sizeof(int32_t), nw, w.f_tok);
            fwrite(bh.data(), sizeof(ggml_fp16_t), bh.size(), w.f_h);
            fwrite(bti.data(), sizeof(int32_t), bti.size(), w.f_ti);
            fwrite(btp.data(), sizeof(ggml_fp16_t), btp.size(), w.f_tp);
            if (!layers.empty()) {
                const size_t L = layers.size();
                baux.resize((size_t) nw * L * n_embd);
                for (size_t k = 0; k < L; k++) {
                    const float * src = llama_get_embeddings_layer_inp(ctx, layers[k]);
                    if (!src) {
                        fprintf(stderr, "layer %u input not extracted (seq %s)\n", layers[k], id.c_str());
                        return 1;
                    }
                    for (int i = i0; i < nb; i++) {
                        ggml_fp32_to_fp16_row(src + (size_t) i * n_embd, baux.data() + ((size_t) (i - i0) * L + k) * n_embd, n_embd);
                    }
                }
                fwrite(baux.data(), sizeof(ggml_fp16_t), baux.size(), w.f_aux);
            }
        }

        const int n_written = n - keep_from;
        json rec = { { "id", id }, { "start", w.n }, { "length", n_written }, { "n_prompt", n_prompt - keep_from }, { "skipped", keep_from } };
        if (j.contains("conv")) {
            rec["conv"] = j["conv"];
        }
        const std::string s = rec.dump() + "\n";
        fwrite(s.data(), 1, s.size(), w.f_idx);
        fflush(w.f_idx);

        w.n         += n_written;
        n_pos_total += n;
        n_seq++;
        if (n_seq % 10 == 0) {
            const double dt = (ggml_time_us() - t0) / 1e6;
            fprintf(stderr, "seqs %lld  positions %lld  %.0f tok/s\n", (long long) n_seq, (long long) n_pos_total, n_pos_total / dt);
        }
    }
    w.close();

    const double dt = (ggml_time_us() - t0) / 1e6;
    fprintf(stderr, "done: %lld sequences, %lld positions, %.1f s, %.0f tok/s, n_embd %d, n_vocab %d\n",
            (long long) n_seq, (long long) n_pos_total, dt, n_pos_total / dt, n_embd, n_vocab);

    llama_batch_ext_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
