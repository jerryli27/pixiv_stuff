from data_util import print_tags_in_database

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_dir", default = "pixiv_1T_db.pkl",
                        help="path to an already existing database or to store the new database.")
    parser.add_argument("--tag_max_count", type=int, default=100,
                        help="number of the most popular tags to be taken into account.")
    a = parser.parse_args()
    print_tags_in_database(a.db_dir, a.tag_max_count)