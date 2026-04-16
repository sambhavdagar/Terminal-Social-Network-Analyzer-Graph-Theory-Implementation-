# Terminal-Social-Network-Analyzer-Graph-Theory-Implementation-

Terminal Social Network Analyzer
A comprehensive, single-file C++17 application designed to model, analyze, and simulate social networks using advanced graph theory algorithms. This project operates as an interactive REPL (Read-Eval-Print Loop) terminal interface, bridging the gap between theoretical graph algorithms and practical software engineering.

🚀 Features & Algorithms Implemented
Centrality & Influence: * PageRank: Computes user influence using power iteration.

Betweenness Centrality: Implemented via Brandes' Algorithm (adapted for weighted edges using Dijkstra's).

Closeness & Degree Centrality: Ranks nodes based on shortest-path reachability and direct connections.

Pathfinding: Utilizes Dijkstra's Algorithm with a priority queue to find the shortest weighted path between two users.

Community Detection & Clustering: * Label Propagation Algorithm to detect natural clusters/communities.

Calculates local and average Clustering Coefficients.

Identifies Connected Components using Depth-First Search (DFS).

Structural Network Analysis: Employs Tarjan's DFS-based algorithms to locate critical Bridges and Articulation Points.

Recommendation Engine: Calculates Jaccard Similarity to recommend potential mutual friends.

Zero-Dependency Data Persistence: Features custom-built parsers to import and export network data in both CSV and JSON formats without external libraries.

🛠️ Technical Stack & Architecture
Language: C++17

Data Structures: * Weighted undirected graph represented via std::unordered_map<Node, std::unordered_map<Node, Weight>> for O(1) average time complexity edge lookups.

Extensive use of std::priority_queue, std::stack, and std::vector for optimized traversals.
