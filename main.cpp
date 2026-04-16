// social_graph.cpp
// Terminal Social Network Analyzer (C++17, single-file)
// Implements: users, weighted connections, mutual friends, jaccard recommender,
// Dijkstra shortest path, PageRank, degree/closeness/betweenness centralities (Brandes),
// bridges, articulation points, clustering coefficient, connected components,
// CSV import/export, JSON import/export (simple), sample graphs, simulation, timing.

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <queue>
#include <stack>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>
#include <limits>
#include <functional>
#include <cctype>
using namespace std;

using Node = string;
using Weight = double;

struct Edge {
    Node u, v;
    Weight w;
    Edge() {}
    Edge(const Node& a, const Node& b, Weight weight) : u(a), v(b), w(weight) {}
};

class SocialNetwork {
private:
    // adjacency: node -> map(neighbor -> weight)
    unordered_map<Node, unordered_map<Node, Weight>> adj;
    bool verbose = false;

    void log(const string &msg) const {
        if (verbose) cerr << "[LOG] " << msg << "\n";
    }

public:
    SocialNetwork(bool v=false) : verbose(v) {}

    // --------------------------
    // Basic node / edge ops
    // --------------------------
    void clear() {
        adj.clear();
    }

    vector<Node> users() const {
        vector<Node> out;
        out.reserve(adj.size());
        for (auto &p : adj) out.push_back(p.first);
        return out;
    }

    bool has_user(const Node &u) const {
        return adj.find(u) != adj.end();
    }

    void add_user(const Node &u) {
        if (!has_user(u)) {
            adj[u] = unordered_map<Node, Weight>();
            log("Added user " + u);
        } else {
            log("User exists: " + u);
        }
    }

    void remove_user(const Node &u) {
        if (!has_user(u)) return;
        // remove node from neighbors
        for (auto &nbr : adj[u]) {
            adj[nbr.first].erase(u);
        }
        adj.erase(u);
        log("Removed user " + u);
    }

    void add_connection(const Node &a, const Node &b, Weight w = 1.0) {
        if (a == b) return;
        if (!has_user(a)) add_user(a);
        if (!has_user(b)) add_user(b);
        if (adj[a].find(b) != adj[a].end()) {
            // update weight to max like Python version
            adj[a][b] = max(adj[a][b], w);
            adj[b][a] = max(adj[b][a], w);
            log("Updated edge " + a + "-" + b);
        } else {
            adj[a][b] = w;
            adj[b][a] = w;
            log("Added edge " + a + "-" + b);
        }
    }

    void remove_connection(const Node &a, const Node &b) {
        if (has_user(a)) adj[a].erase(b);
        if (has_user(b)) adj[b].erase(a);
    }

    vector<pair<Node, Weight>> connections(const Node &u) const {
        vector<pair<Node, Weight>> out;
        if (!has_user(u)) return out;
        for (auto &p : adj.at(u)) out.emplace_back(p.first, p.second);
        return out;
    }

    // --------------------------
    // Mutual friends
    // --------------------------
    vector<Node> mutual_friends(const Node &a, const Node &b) const {
        vector<Node> out;
        if (!has_user(a) || !has_user(b)) return out;
        unordered_set<Node> sa;
        for (auto &p : adj.at(a)) sa.insert(p.first);
        for (auto &p : adj.at(b)) if (sa.count(p.first)) out.push_back(p.first);
        return out;
    }

    // --------------------------
    // Jaccard-based recommender
    // --------------------------
    vector<pair<Node, double>> recommend_jaccard(const Node &u, int top_k=5) const {
        vector<pair<Node,double>> out;
        if (!has_user(u)) return out;
        unordered_set<Node> u_neigh;
        for (auto &p : adj.at(u)) u_neigh.insert(p.first);

        for (auto &p : adj) {
            const Node &v = p.first;
            if (v == u) continue;
            if (u_neigh.count(v)) continue;
            unordered_set<Node> v_neigh;
            for (auto &q : p.second) v_neigh.insert(q.first);
            size_t inter = 0;
            for (const Node &x : u_neigh) if (v_neigh.count(x)) inter++;
            size_t uni = u_neigh.size() + v_neigh.size() - inter;
            double score = (uni>0) ? double(inter)/double(uni) : 0.0;
            if (score > 0.0) out.emplace_back(v, score);
        }
        sort(out.begin(), out.end(), [](auto &a, auto &b){ return a.second > b.second; });
        if ((int)out.size() > top_k) out.resize(top_k);
        return out;
    }

    // --------------------------
    // Dijkstra's shortest path (weights)
    // --------------------------
    pair<vector<Node>, double> shortest_path_dijkstra(const Node &src, const Node &dst) const {
        if (!has_user(src) || !has_user(dst)) return {{}, 0.0};
        using P = pair<double, Node>; // dist, node
        unordered_map<Node, double> dist;
        unordered_map<Node, Node> parent;
        for (auto &p : adj) dist[p.first] = numeric_limits<double>::infinity();
        dist[src] = 0.0;
        priority_queue<P, vector<P>, greater<P>> pq;
        pq.push(make_pair(0.0, src));
        while (!pq.empty()) {
            P top = pq.top(); pq.pop();
            double d = top.first;
            Node u = top.second;
            if (d > dist[u]) continue;
            if (u == dst) break;
            for (auto &kv : adj.at(u)) {
                const Node &v = kv.first;
                double w = kv.second;
                if (dist[v] > dist[u] + w) {
                    dist[v] = dist[u] + w;
                    parent[v] = u;
                    pq.push(make_pair(dist[v], v));
                }
            }
        }
        if (dist[dst] == numeric_limits<double>::infinity()) return {{}, 0.0};
        vector<Node> path;
        Node cur = dst;
        while (cur != src) {
            path.push_back(cur);
            cur = parent[cur];
        }
        path.push_back(src);
        reverse(path.begin(), path.end());
        return {path, dist[dst]};
    }

    // --------------------------
    // PageRank (power iteration)
    // --------------------------
    unordered_map<Node, double> influence_pagerank(double alpha=0.85, int max_iter=100, double tol=1e-6) const {
        unordered_map<Node,double> pr;
        vector<Node> nodes;
        for (auto &p : adj) nodes.push_back(p.first);
        int n = nodes.size();
        if (n == 0) return pr;
        unordered_map<Node,int> idx;
        for (int i=0;i<n;++i) idx[nodes[i]] = i;
        vector<double> rank(n, 1.0/double(n)), next_rank(n, 0.0);
        // Precompute out-degree weights sum
        vector<double> outw(n, 0.0);
        for (int i=0;i<n;++i) {
            for (auto &q : adj.at(nodes[i])) outw[i] += q.second;
        }
        for (int iter=0; iter<max_iter; ++iter) {
            fill(next_rank.begin(), next_rank.end(), (1.0-alpha)/double(n));
            for (int i=0;i<n;++i) {
                if (outw[i] == 0.0) continue;
                for (auto &q : adj.at(nodes[i])) {
                    int j = idx.at(q.first);
                    // distribute proportional to weight
                    next_rank[j] += alpha * rank[i] * (q.second / outw[i]);
                }
            }
            double diff = 0.0;
            for (int i=0;i<n;++i) {
                diff += fabs(next_rank[i] - rank[i]);
                rank[i] = next_rank[i];
            }
            if (diff < tol) break;
        }
        for (int i=0;i<n;++i) pr[nodes[i]] = rank[i];
        return pr;
    }

    // --------------------------
    // Degree, Closeness centrality
    // --------------------------
    unordered_map<Node, double> degree_centrality() const {
        unordered_map<Node, double> out;
        int n = adj.size();
        if (n <= 1) {
            for (auto &p : adj) out[p.first] = 0.0;
            return out;
        }
        for (auto &p : adj) out[p.first] = double(p.second.size()) / double(n-1);
        return out;
    }

    unordered_map<Node, double> closeness_centrality() const {
        unordered_map<Node, double> out;
        int n = adj.size();
        for (auto &p : adj) {
            const Node &source = p.first;
            // BFS / Dijkstra since edges weighted: use Dijkstra to compute shortest path distances
            unordered_map<Node,double> dist;
            for (auto &q : adj) dist[q.first] = numeric_limits<double>::infinity();
            dist[source] = 0.0;
            using P = pair<double, Node>;
        priority_queue<P, vector<P>, greater<P>> pq;
        pq.push(make_pair(0.0, source));
        while (!pq.empty()) {
            P top = pq.top(); pq.pop();
            double d = top.first;
            Node u = top.second;
            if (d > dist[u]) continue;
            for (auto &kv : adj.at(u)) {
                if (dist[kv.first] > dist[u] + kv.second) {
                    dist[kv.first] = dist[u] + kv.second;
                    pq.push(make_pair(dist[kv.first], kv.first));
                    }
                }
            }
            double sumd = 0.0;
            int reachable = 0;
            for (auto &r : dist) if (r.second < numeric_limits<double>::infinity()) { sumd += r.second; ++reachable; }
            if (sumd > 0.0 && reachable > 1) out[source] = double(reachable-1) / sumd;
            else out[source] = 0.0;
        }
        return out;
    }

    // --------------------------
    // Betweenness centrality (Brandes algorithm)
    // Weighted version (edge weights)
    // --------------------------
    unordered_map<Node, double> betweenness_centrality() const {
        unordered_map<Node, double> CB;
        for (auto &p : adj) CB[p.first] = 0.0;

        // For weighted graphs, use Dijkstra-based Brandes
        for (auto &s_pair : adj) {
            Node s = s_pair.first;
            // stack for order
            vector<Node> S;
            // predecessors
            unordered_map<Node, vector<Node>> P;
            // sigma: number of shortest paths
            unordered_map<Node, double> sigma;
            // distance
            unordered_map<Node, double> dist;
            for (auto &p : adj) {
                P[p.first] = {};
                sigma[p.first] = 0.0;
                dist[p.first] = numeric_limits<double>::infinity();
            }
            sigma[s] = 1.0;
            dist[s] = 0.0;
            using PDI = pair<double, Node>;
            priority_queue<PDI, vector<PDI>, greater<PDI>> Q;
            Q.push(make_pair(0.0, s));
            while (!Q.empty()) {
                PDI top = Q.top(); Q.pop();
                double d = top.first;
                Node v = top.second;
                if (d > dist[v]) continue;
                S.push_back(v);
                for (auto &edge : adj.at(v)) {
                    Node w = edge.first;
                    double vw_weight = edge.second;
                    double nd = dist[v] + vw_weight;
                    if (dist[w] > nd) {
                        dist[w] = nd;
                        Q.push(make_pair(nd, w));
                        sigma[w] = sigma[v];
                        P[w].clear();
                        P[w].push_back(v);
                    } else if (fabs(dist[w] - nd) < 1e-12) {
                        sigma[w] += sigma[v];
                        P[w].push_back(v);
                    }
                }
            }
            unordered_map<Node, double> delta;
            for (auto &p : adj) delta[p.first] = 0.0;
            // S returns vertices in non-decreasing distance; process in reverse
            for (int i = (int)S.size()-1; i>=0; --i) {
                Node w = S[i];
                for (Node v : P[w]) {
                    if (sigma[w] != 0)
                        delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
                }
                if (w != s) CB[w] += delta[w];
            }
        }
        // normalize for undirected graphs: divide by 2
        for (auto &p : CB) p.second /= 2.0;
        return CB;
    }

    // --------------------------
    // Bridges and articulation points (Tarjan)
    // --------------------------
    vector<pair<Node, Node>> find_bridges() const {
        vector<pair<Node, Node>> bridges;
        unordered_map<Node, int> disc, low;
        unordered_map<Node, bool> visited;
        int time_dfs = 0;

        function<void(const Node&, const Node&)> dfs = [&](const Node &u, const Node &parent){
            visited[u] = true;
            disc[u] = low[u] = ++time_dfs;
            for (auto &kv : adj.at(u)) {
                Node v = kv.first;
                if (!visited[v]) {
                    dfs(v, u);
                    low[u] = min(low[u], low[v]);
                    if (low[v] > disc[u]) {
                        bridges.emplace_back(u, v);
                    }
                } else if (v != parent) {
                    low[u] = min(low[u], disc[v]);
                }
            }
        };

        for (auto &p : adj) {
            if (!visited[p.first]) dfs(p.first, "");
        }
        return bridges;
    }

    vector<Node> find_articulation_points() const {
        unordered_set<Node> aps;
        unordered_map<Node, int> disc, low;
        unordered_map<Node, bool> visited;
        int time_dfs = 0;

        function<void(const Node&, const Node&)> dfs = [&](const Node &u, const Node &parent){
            visited[u] = true;
            disc[u] = low[u] = ++time_dfs;
            int children = 0;
            for (auto &kv : adj.at(u)) {
                Node v = kv.first;
                if (!visited[v]) {
                    children++;
                    dfs(v, u);
                    low[u] = min(low[u], low[v]);
                    if (parent != "" && low[v] >= disc[u]) aps.insert(u);
                    if (parent == "" && children > 1) aps.insert(u);
                } else if (v != parent) {
                    low[u] = min(low[u], disc[v]);
                }
            }
        };

        for (auto &p : adj) {
            if (!visited[p.first]) dfs(p.first, "");
        }

        vector<Node> out(aps.begin(), aps.end());
        return out;
    }

    // --------------------------
    // Clustering coefficients
    // (local and average)
    // --------------------------
    double clustering_coeff_local(const Node &u) const {
        if (!has_user(u)) return 0.0;
        int deg = adj.at(u).size();
        if (deg < 2) return 0.0;
        int tri = 0;
        for (auto &a : adj.at(u)) {
            for (auto &b : adj.at(u)) {
                if (a.first >= b.first) continue; // avoid double count
                if (adj.at(a.first).find(b.first) != adj.at(a.first).end()) tri++;
            }
        }
        double possible = double(deg) * (deg - 1) / 2.0;
        return double(tri) / possible;
    }

    double clustering_coeff_average() const {
        if (adj.empty()) return 0.0;
        double sum = 0.0;
        for (auto &p : adj) sum += clustering_coeff_local(p.first);
        return sum / double(adj.size());
    }

    // --------------------------
    // Graph stats & components
    // --------------------------
    struct Stats {
        int num_nodes;
        int num_edges;
        double avg_clustering;
    };

    Stats graph_stats() const {
        int n = adj.size();
        int m = 0;
        for (auto &p : adj) m += p.second.size();
        m /= 2;
        double avg_cl = clustering_coeff_average();
        return {n,m,avg_cl};
    }

    vector<vector<Node>> connected_components() const {
        vector<vector<Node>> comps;
        unordered_map<Node, bool> vis;
        for (auto &p : adj) vis[p.first] = false;
        for (auto &p : adj) {
            if (vis[p.first]) continue;
            vector<Node> comp;
            stack<Node> st;
            st.push(p.first);
            vis[p.first] = true;
            while (!st.empty()) {
                Node u = st.top(); st.pop();
                comp.push_back(u);
                for (auto &kv : adj.at(u)) {
                    if (!vis[kv.first]) {
                        vis[kv.first] = true;
                        st.push(kv.first);
                    }
                }
            }
            comps.push_back(comp);
        }
        return comps;
    }

    // --------------------------
    // Community detection - simple label propagation
    // --------------------------
    vector<unordered_set<Node>> communities_label_propagation(int max_iter=50) const {
        vector<Node> nodes;
        for (auto &p : adj) nodes.push_back(p.first);
        int n = nodes.size();
        unordered_map<Node, Node> label;
        for (auto &v : nodes) label[v] = v;
        for (int iter=0; iter<max_iter; ++iter) {
            bool changed = false;
            // random order
            vector<int> order(n);
            iota(order.begin(), order.end(), 0);
            shuffle(order.begin(), order.end(), std::mt19937{std::random_device{}()});
            for (int idx : order) {
                Node v = nodes[idx];
                unordered_map<Node, int> counts;
                for (auto &kv : adj.at(v)) counts[label.at(kv.first)]++;
                if (counts.empty()) continue;
                // choose label with max count
                Node best = label[v];
                int bestc = -1;
                for (auto &kv : counts) {
                    if (kv.second > bestc || (kv.second==bestc && kv.first < best)) {
                        best = kv.first; bestc = kv.second;
                    }
                }
                if (best != label[v]) { label[v] = best; changed = true; }
            }
            if (!changed) break;
        }
        unordered_map<Node, unordered_set<Node>> groups;
        for (auto &p : label) groups[p.second].insert(p.first);
        vector<unordered_set<Node>> out;
        for (auto &g : groups) out.push_back(g.second);
        return out;
    }

    // --------------------------
    // CSV import/export (edges)
    // Each row: u,v,weight
    // --------------------------
    bool import_csv(const string &path) {
        ifstream fin(path);
        if (!fin.is_open()) return false;
        string line;
        while (getline(fin, line)) {
            if (line.empty()) continue;
            stringstream ss(line);
            string a,b,wstr;
            if (!getline(ss, a, ',')) continue;
            if (!getline(ss, b, ',')) continue;
            if (!getline(ss, wstr)) wstr = "1.0";
            // trim
            auto trim = [](string &s){ while(!s.empty() && isspace(s.back())) s.pop_back(); while(!s.empty() && isspace(s.front())) s.erase(s.begin()); };
            trim(a); trim(b); trim(wstr);
            double w = 1.0;
            try { w = stod(wstr); } catch(...) { w = 1.0; }
            add_connection(a, b, w);
        }
        return true;
    }

    bool export_csv(const string &path) const {
        ofstream fout(path);
        if (!fout.is_open()) return false;
        unordered_set<string> seen;
        for (auto &p : adj) {
            for (auto &q : p.second) {
                string a = p.first, b = q.first;
                // ensure each undirected edge written once (lexicographic guard)
                if (a < b) fout << a << "," << b << "," << q.second << "\n";
            }
        }
        return true;
    }

    // --------------------------
    // JSON import/export (simple)
    // Format:
    // {"nodes":["A","B",...],"edges":[["A","B",1.2],["B","C",1.0]]}
    // --------------------------
    bool save_json(const string &path) const {
        ofstream fout(path);
        if (!fout.is_open()) return false;
        fout << "{\n";
        fout << "  \"nodes\": [";
        bool first = true;
        for (auto &p : adj) {
            if (!first) fout << ", ";
            fout << "\"" << escape_json(p.first) << "\"";
            first = false;
        }
        fout << "],\n";
        fout << "  \"edges\": [\n";
        bool firstEdge = true;
        for (auto &p : adj) {
            for (auto &q : p.second) {
                if (p.first < q.first) {
                    if (!firstEdge) fout << ",\n";
                    fout << "    [\"" << escape_json(p.first) << "\", \"" << escape_json(q.first) << "\", " << q.second << "]";
                    firstEdge = false;
                }
            }
        }
        fout << "\n  ]\n}\n";
        return true;
    }

    bool load_json(const string &path) {
        ifstream fin(path);
        if (!fin.is_open()) return false;
        // Note: simple parser, expects the format we write above. This is not a full JSON parser.
        string content((istreambuf_iterator<char>(fin)), istreambuf_iterator<char>());
        // find nodes array
        clear();
        size_t nodes_pos = content.find("\"nodes\"");
        if (nodes_pos != string::npos) {
            size_t start = content.find('[', nodes_pos);
            size_t end = content.find(']', start);
            if (start != string::npos && end != string::npos && end > start) {
                string nodes_block = content.substr(start+1, end-start-1);
                vector<string> nodes = split_and_unquote(nodes_block, ',');
                for (auto &n : nodes) {
                    if (!n.empty()) add_user(unescape_json(n));
                }
            }
        }
        size_t edges_pos = content.find("\"edges\"");
        if (edges_pos != string::npos) {
            size_t start = content.find('[', edges_pos);
            size_t end = content.find(']', start);
            // edges array spans potentially multiple lines; find the matching closing bracket
            if (start != string::npos) {
                // find the matching closing bracket for edges array
                int depth = 0;
                size_t i = start;
                for (; i < content.size(); ++i) {
                    if (content[i] == '[') depth++;
                    else if (content[i] == ']') { depth--; if (depth==0) break; }
                }
                if (i < content.size()) {
                    string edges_block = content.substr(start+1, i-start-1);
                    // parse each triple: ["A","B",1.2]
                    size_t pos = 0;
                    while (true) {
                        size_t open = edges_block.find('[', pos);
                        if (open == string::npos) break;
                        size_t close = edges_block.find(']', open);
                        if (close == string::npos) break;
                        string triple = edges_block.substr(open+1, close-open-1);
                        // triple elements separated by commas, but strings quoted
                        vector<string> parts = split_and_unquote(triple, ',');
                        if (parts.size() >= 2) {
                            string a = unescape_json(parts[0]);
                            string b = unescape_json(parts[1]);
                            double w = 1.0;
                            if (parts.size() > 2) {
                                try { w = stod(parts[2]); } catch(...) { w = 1.0; }
                            }
                            add_connection(a,b,w);
                        }
                        pos = close+1;
                    }
                }
            }
        }
        return true;
    }

    // --------------------------
    // Sample graphs and simulation
    // --------------------------
    void sample_graph() {
        clear();
        vector<string> users = {"Alice","Bob","Carol","Dave","Eve","Frank","Grace"};
        for (auto &u : users) add_user(u);
        add_connection("Alice","Bob",1.0);
        add_connection("Alice","Carol",2.0);
        add_connection("Bob","Dave",1.5);
        add_connection("Carol","Dave",2.5);
        add_connection("Dave","Eve",1.0);
        add_connection("Eve","Frank",0.5);
        add_connection("Frank","Grace",1.2);
        add_connection("Carol","Grace",0.8);
    }

    void sample_graph_large(int n=50, int m=100) {
        clear();
        for (int i=0;i<n;++i) add_user("User" + to_string(i));
        random_device rd;
        mt19937 rng(rd());
        uniform_int_distribution<int> uni(0,n-1);
        uniform_real_distribution<double> wdist(0.1,5.0);
        for (int i=0;i<m;++i) {
            int a = uni(rng), b = uni(rng);
            if (a==b) continue;
            add_connection("User" + to_string(a), "User" + to_string(b), round_double(wdist(rng), 2));
        }
    }

    void simulate_random_interactions(int rounds=20) {
        vector<Node> nodes = users();
        if (nodes.empty()) return;
        random_device rd;
        mt19937 rng(rd());
        uniform_int_distribution<int> uni(0, (int)nodes.size()-1);
        uniform_real_distribution<double> wdist(0.5, 3.0);
        for (int i=0;i<rounds;++i) {
            int a = uni(rng), b = uni(rng);
            if (a==b) continue;
            add_connection(nodes[a], nodes[b], wdist(rng));
        }
    }

    // --------------------------
    // Timing helper
    // --------------------------
    template<typename F, typename... Args>
    pair<typename std::result_of<F(Args...)>::type, double> time_method(F&& f, Args&&... args) {
        using Ret = typename std::result_of<F(Args...)>::type;
        auto start = chrono::high_resolution_clock::now();
        Ret out = f(std::forward<Args>(args)...);
        auto end = chrono::high_resolution_clock::now();
        double ms = chrono::duration<double>(end - start).count();
        return {out, ms};
    }

private:
    // --------------------------
    // small helpers for JSON parse/write
    // --------------------------
    static string escape_json(const string &s) {
        string out;
        for (char c : s) {
            if (c == '\\') out += "\\\\";
            else if (c == '\"') out += "\\\"";
            else if (c == '\n') out += "\\n";
            else out.push_back(c);
        }
        return out;
    }
    static string unescape_json(const string &s) {
        string out;
        for (size_t i=0;i<s.size();++i) {
            if (s[i] == '\\' && i+1 < s.size()) {
                char c = s[i+1];
                if (c=='\\' || c=='\"') out.push_back(c), ++i;
                else if (c=='n') out.push_back('\n'), ++i;
                else out.push_back(s[i]);
            } else out.push_back(s[i]);
        }
        // strip surrounding quotes if present
        if (!out.empty() && out.front()=='\"' && out.back()=='\"') return out.substr(1, out.size()-2);
        return out;
    }
    static vector<string> split_and_unquote(const string &s, char delim) {
        vector<string> parts;
        string cur;
        bool inquote = false;
        for (size_t i=0;i<s.size();++i) {
            char c = s[i];
            if (c == '\"') { inquote = !inquote; cur.push_back(c); }
            else if (c == delim && !inquote) {
                string t = trim_copy(cur);
                // strip quotes
                if (!t.empty() && t.front()=='\"' && t.back()=='\"') t = t.substr(1,t.size()-2);
                parts.push_back(t);
                cur.clear();
            } else cur.push_back(c);
        }
        if (!cur.empty()) {
            string t = trim_copy(cur);
            if (!t.empty() && t.front()=='\"' && t.back()=='\"') t = t.substr(1,t.size()-2);
            parts.push_back(t);
        }
        return parts;
    }
    static string trim_copy(const string &s) {
        size_t i=0, j=s.size();
        while (i<j && isspace((unsigned char)s[i])) ++i;
        while (j>i && isspace((unsigned char)s[j-1])) --j;
        return s.substr(i, j-i);
    }
    static double round_double(double x, int prec=2) {
        double m = pow(10.0, prec);
        return floor(x*m + 0.5)/m;
    }
};

// --------------------------
// Terminal UI helpers
// --------------------------
void print_help() {
    cout << "Commands (type the command name):\n";
    cout << " add_user <name>\n remove_user <name>\n add_edge <u> <v> [weight]\n remove_edge <u> <v>\n";
    cout << " list_users\n connections <user>\n mutual <u> <v>\n recommend <user> [k]\n";
    cout << " dijkstra <src> <dst>\n pagerank\n degree\n closeness\n betweenness\n bridges\n aps\n clustering [user]\n stats\n components\n communities\n";
    cout << " sample_small\n sample_large <n> <m>\n simulate <rounds>\n save_json <path>\n load_json <path>\n save_csv <path>\n load_csv <path>\n clear\n help\n exit\n";
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    SocialNetwork G(false);
    cout << "Social Network Analyzer (C++ terminal)\n";
    print_help();
    cout.flush();
    string line;
    while (true) {
        cout << "\n> ";
        cout.flush();
        if (!getline(cin, line)) {
            if (cin.eof()) {
                cout << "\nEOF detected. Exiting.\n";
            } else if (cin.fail()) {
                cout << "\nInput error. Exiting.\n";
            }
            break;
        }
        if (line.empty()) continue;
        // simple tokenize
        stringstream ss(line);
        string cmd; ss >> cmd;
        if (cmd == "help") print_help();
        else if (cmd == "exit") break;
        else if (cmd == "add_user") {
            string u; if (!(ss >> u)) { cout << "Usage: add_user <name>\n"; continue; }
            G.add_user(u); cout << "Added user: " << u << "\n";
        } else if (cmd == "remove_user") {
            string u; if (!(ss >> u)) { cout << "Usage: remove_user <name>\n"; continue; }
            G.remove_user(u); cout << "Removed user: " << u << "\n";
        } else if (cmd == "add_edge") {
            string a,b; double w=1.0; if (!(ss >> a >> b)) { cout<<"Usage: add_edge <u> <v> [weight]\n"; continue; }
            if (ss >> w) {}
            G.add_connection(a,b,w); cout << "Added/updated edge: " << a << " - " << b << " (" << w << ")\n";
        } else if (cmd == "remove_edge") {
            string a,b; if (!(ss >> a >> b)) { cout<<"Usage: remove_edge <u> <v>\n"; continue; }
            G.remove_connection(a,b); cout << "Removed edge: " << a << " - " << b << "\n";
        } else if (cmd == "list_users") {
            auto u = G.users();
            cout << "Users ("<<u.size()<<"):\n";
            for (auto &x : u) cout << "  " << x << "\n";
        } else if (cmd == "connections") {
            string u; if (!(ss >> u)) { cout<<"Usage: connections <user>\n"; continue; }
            auto con = G.connections(u);
            cout << "Connections of " << u << " ("<<con.size()<<"):\n";
            for (auto &kv : con) cout << "  " << kv.first << " (w=" << kv.second << ")\n";
        } else if (cmd == "mutual") {
            string a,b; if (!(ss >> a >> b)) { cout<<"Usage: mutual <u> <v>\n"; continue; }
            auto mf = G.mutual_friends(a,b);
            cout << "Mutual friends of " << a << " and " << b << ":\n";
            for (auto &x : mf) cout << "  " << x << "\n";
        } else if (cmd == "recommend") {
            string u; int k=5; if (!(ss >> u)) { cout<<"Usage: recommend <user> [k]\n"; continue; }
            ss >> k;
            auto rec = G.recommend_jaccard(u,k);
            cout << "Recommendations for " << u << ":\n";
            for (auto &p : rec) cout << "  " << p.first << " (score="<<p.second<<")\n";
        } else if (cmd == "dijkstra") {
            string a,b; if (!(ss >> a >> b)) { cout<<"Usage: dijkstra <src> <dst>\n"; continue; }
            auto res = G.shortest_path_dijkstra(a,b);
            if (res.first.empty()) cout << "No path found.\n";
            else {
                cout << "Distance: " << res.second << "\nPath: ";
                for (size_t i=0;i<res.first.size();++i) {
                    if (i) cout << " -> ";
                    cout << res.first[i];
                }
                cout << "\n";
            }
        } else if (cmd == "pagerank") {
            auto pr = G.influence_pagerank();
            cout << "PageRank scores:\n";
            vector<pair<string,double>> v; for (auto &p: pr) v.push_back(p);
            sort(v.begin(), v.end(), [](auto &a, auto &b){ return a.second > b.second; });
            for (auto &p : v) cout << "  " << p.first << " : " << p.second << "\n";
        } else if (cmd == "degree") {
            auto dc = G.degree_centrality();
            cout << "Degree centrality:\n";
            vector<pair<string,double>> v; for (auto &p: dc) v.push_back(p);
            sort(v.begin(), v.end(), [](auto &a, auto &b){ return a.second > b.second; });
            for (auto &p : v) cout << "  " << p.first << " : " << p.second << "\n";
        } else if (cmd == "closeness") {
            auto cc = G.closeness_centrality();
            cout << "Closeness centrality:\n";
            vector<pair<string,double>> v; for (auto &p: cc) v.push_back(p);
            sort(v.begin(), v.end(), [](auto &a, auto &b){ return a.second > b.second; });
            for (auto &p : v) cout << "  " << p.first << " : " << p.second << "\n";
        } else if (cmd == "betweenness") {
            cout << "Computing betweenness centrality (may take time)...\n";
            auto res = G.betweenness_centrality();
            vector<pair<string,double>> v; for (auto &p : res) v.push_back(p);
            sort(v.begin(), v.end(), [](auto &a, auto &b){ return a.second > b.second; });
            for (auto &p : v) cout << "  " << p.first << " : " << p.second << "\n";
        } else if (cmd == "bridges") {
            auto br = G.find_bridges();
            cout << "Bridges ("<<br.size()<<"):\n";
            for (auto &e : br) cout << "  " << e.first << " - " << e.second << "\n";
        } else if (cmd == "aps") {
            auto aps = G.find_articulation_points();
            cout << "Articulation points ("<<aps.size()<<"):\n";
            for (auto &x : aps) cout << "  " << x << "\n";
        } else if (cmd == "clustering") {
            string maybe; if (ss >> maybe) {
                cout << "Local clustering coeff for " << maybe << ": " << G.clustering_coeff_local(maybe) << "\n";
            } else {
                cout << "Average clustering coefficient: " << G.clustering_coeff_average() << "\n";
            }
        } else if (cmd == "stats") {
            auto s = G.graph_stats();
            cout << "Nodes: " << s.num_nodes << "  Edges: " << s.num_edges << "  Avg clustering: " << s.avg_clustering << "\n";
        } else if (cmd == "components") {
            auto comps = G.connected_components();
            cout << "Connected components ("<<comps.size()<<"):\n";
            for (auto &c : comps) {
                cout << "  [";
                for (size_t i=0;i<c.size();++i) {
                    if (i) cout << ", ";
                    cout << c[i];
                }
                cout << "]\n";
            }
        } else if (cmd == "communities") {
            auto cs = G.communities_label_propagation();
            cout << "Communities (label-propagation) ("<<cs.size()<<"):\n";
            for (auto &c : cs) {
                cout << "  [";
                bool first = true;
                for (auto &n : c) {
                    if (!first) cout << ", ";
                    cout << n;
                    first = false;
                }
                cout << "]\n";
            }
        } else if (cmd == "sample_small") {
            G.sample_graph(); cout << "Sample small graph loaded.\n";
        } else if (cmd == "sample_large") {
            int n=50, m=100; ss >> n >> m;
            G.sample_graph_large(n,m); cout << "Sample large graph loaded.\n";
        } else if (cmd == "simulate") {
            int rounds = 20; ss >> rounds;
            G.simulate_random_interactions(rounds); cout << "Simulated " << rounds << " interactions.\n";
        } else if (cmd == "save_json") {
            string path; if (!(ss >> path)) { cout<<"Usage: save_json <path>\n"; continue; }
            if (G.save_json(path)) cout<<"Saved JSON to "<<path<<"\n"; else cout<<"Failed to save JSON\n";
        } else if (cmd == "load_json") {
            string path; if (!(ss >> path)) { cout<<"Usage: load_json <path>\n"; continue; }
            if (G.load_json(path)) cout<<"Loaded JSON from "<<path<<"\n"; else cout<<"Failed to load JSON\n";
        } else if (cmd == "save_csv") {
            string path; if (!(ss >> path)) { cout<<"Usage: save_csv <path>\n"; continue; }
            if (G.export_csv(path)) cout<<"Saved CSV to "<<path<<"\n"; else cout<<"Failed to save CSV\n";
        } else if (cmd == "load_csv") {
            string path; if (!(ss >> path)) { cout<<"Usage: load_csv <path>\n"; continue; }
            if (G.import_csv(path)) cout<<"Loaded CSV from "<<path<<"\n"; else cout<<"Failed to load CSV\n";
        } else if (cmd == "clear") {
            G.clear(); cout << "Graph cleared.\n";
        } else {
            cout << "Unknown command. Type 'help' to see commands.\n";
        }
    }
    cout << "Bye.\n";
    cout.flush();
    // Pause before exiting (useful when double-clicking the .exe on Windows)
    #ifdef _WIN32
    system("pause");
    #endif
    return 0;
}
