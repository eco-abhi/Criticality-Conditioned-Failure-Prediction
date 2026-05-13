import pandas as pd
import pickle
import networkx as nx 

# Part catalog
pc = pd.read_csv("outputs/part_catalog.csv")
print("=== PART CATALOG ===")
print(pc.shape)
print(pc["criticality_class"].value_counts(normalize=True).round(3))
print(pc[["lead_time_mean_weeks","lead_time_cv","n_qualified_suppliers"]].describe().round(3))

# Supplier history
sh = pd.read_csv("outputs/supplier_history.csv")
print("\n=== SUPPLIER HISTORY ===")
print(sh.shape)
print(sh.groupby("supplier_at_risk_flag")["otd_oem_measured"].mean().round(3))

# Compliance outcomes
co = pd.read_csv("outputs/compliance_outcomes.csv")
print("\n=== COMPLIANCE OUTCOMES ===")
print(co.shape)
print(co.groupby("criticality_class")["compliance_failure"].mean().round(3))
print("Overall failure rate:", co["compliance_failure"].mean().round(3))

# BOM graph
with open("outputs/bom_graph.gpickle","rb") as f:
    G = pickle.load(f)
print("\n=== BOM GRAPH ===")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
print("Is DAG:", __import__("networkx").is_directed_acyclic_graph(G))

sh = pd.read_csv("outputs/supplier_history.csv")
print(sh[["supplier_at_risk_flag","otd_supplier_reported","otd_oem_measured"]].groupby("supplier_at_risk_flag").mean().round(3))


with open("outputs/bom_graph.gpickle", "rb") as f:
    G = pickle.load(f)

isolated = list(nx.isolates(G))
weakly_connected = list(nx.weakly_connected_components(G))

print(f"Isolated nodes (no edges at all): {len(isolated)}")
print(f"Number of weakly connected components: {len(weakly_connected)}")
print(f"Largest component size: {max(len(c) for c in weakly_connected)}")
print(f"Nodes with in-degree 0: {sum(1 for _, d in G.in_degree() if d == 0)}")
print(f"Nodes with out-degree 0: {sum(1 for _, d in G.out_degree() if d == 0)}")