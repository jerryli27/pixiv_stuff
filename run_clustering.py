import data_util
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="path to folder containing images")
    parser.add_argument("--db_dir", default = "pixiv_1T_db.pkl",
                        help="path to an already existing database or to store the new database.")
    parser.add_argument("--output_dir", default = "pixiv_1T_clustered/", help="output path")
    parser.add_argument("--tag_max_count", type=int, default=10000,
                        help="number of the most popular tags to be taken into account.")
    parser.add_argument("--num_clusters", type=int, default=20, help="Number of clusters to generate.")
    a = parser.parse_args()
    data_util.cluster_images(a.input_dir, a.db_dir, a.output_dir, a.tag_max_count, a.num_clusters)

"""
python run_clustering.py --input_dir=/mnt/pixiv_drive/home/ubuntu/PycharmProjects/PixivUtil2/pixiv_downloaded/
# 25000 is using up 64% of the 64 gig memory
python run_clustering.py --input_dir=/mnt/pixiv_drive/home/ubuntu/PycharmProjects/PixivUtil2/pixiv_downloaded/ --tag_max_count=25000 --num_clusters=100 --output_dir=pixiv_1T_clustered_25000_tags_100_clusters
"""