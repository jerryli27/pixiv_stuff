import data_util
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default = "pixiv_1T_clustered/", help="output path")
    parser.add_argument("--num_clusters", type=int, default=20, help="Number of clusters to generate.")
    parser.add_argument("--num_sample_per_cluster", type=int, default=40, help="Number of samples to save for each cluster.")
    a = parser.parse_args()
    data_util.create_cluster_sample(a.output_dir, a.num_clusters, a.num_sample_per_cluster)

"""
python run_cluster_sampling.py --num_clusters=100 --output_dir=pixiv_1T_clustered_25000_tags_100_clusters_k_means
"""