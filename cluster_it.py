import retworkx as rx
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--csv', type=str, required=True, help="pairwise csv file")
parser.add_argument('--cutoff', type=int, required=True,
                    help="clustering threshold (0:100)%")
parser.add_argument('--output', type=str, required=True, help="clusters output")

args = parser.parse_args()

pairwise_file = args.csv
CONTAINMENT_THRESHOLD = args.cutoff
output = args.output


distance_col_idx = 2


vertices_set = set()
no_lines = 0
print("Parsing vertices...")
with open(pairwise_file) as IN:
    next(IN)
    for line in IN:
        genome_1, genome_2 = tuple(line.strip().split(',')[:2])
        vertices_set.add(genome_1)
        vertices_set.add(genome_2)
        no_lines += 1

print(f"parsed {len(vertices_set)} vertices.")

name_to_id = {}
id_to_name = {}

for i, genome_str in enumerate(vertices_set):
    name_to_id[genome_str] = i
    id_to_name[i] = genome_str

del vertices_set

graph = rx.PyGraph()
nodes_indeces = graph.add_nodes_from(list(id_to_name.keys()))

batch_size = 10000000
batch_counter = 0
edges_tuples = []

print("[i] constructing graph")
with open(pairwise_file, 'r') as pairwise_tsv:
    next(pairwise_tsv)  # skip header
    for row in tqdm(pairwise_tsv, total=no_lines):
        row = row.strip().split(',')
        seq1 = name_to_id[row[0]]
        seq2 = name_to_id[row[1]]
        distance = float(row[distance_col_idx]) * 100

        # don't make graph edge
        if distance < CONTAINMENT_THRESHOLD:
            continue

        if batch_counter < batch_size:
            batch_counter += 1
            edges_tuples.append((seq1, seq2, distance))
        else:
            graph.add_edges_from(edges_tuples)
            batch_counter = 0
            edges_tuples.clear()

    else:
        if len(edges_tuples):
            graph.add_edges_from(edges_tuples)

print("clustering...")
connected_components = rx.connected_components(graph)
print(f"number of clusters: {len(connected_components)}")
print("printing results")
single_components = 0
with open(output, 'w') as CLUSTERS:
    for component in connected_components:
        # uncomment to exclude single genome clusters from exporting
        # if len(component) == 1:
        #     single_components += 1
        #     continue
        named_component = [id_to_name[node] for node in component]
        CLUSTERS.write(','.join(named_component) + '\n')

# print(f"skipped clusters with single node: {single_components}")
