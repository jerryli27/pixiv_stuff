#!/usr/bin/env python
# -*- coding: utf-8 -*-
import shutil
import tempfile
import unittest
import datetime
import time
from sklearn.random_projection import sparse_random_matrix

from data_util import *

PIXIV_INFO_CONTENT = u"""ArtistID   = 163536
ArtistName = Rella
ImageID    = 33703665
Title      = glow
Caption    = きっと このまま君を 溶かして夜になるだけ
Tags       = 初音ミク, 素晴らしい, 夕焼け, ふつくしい, クリック推奨, SAIの本気, VOCALOID, メイキング希望!, これはいい初音, VOCALOID50000users入り
Comments   = comment_author_id: 22917715; comment: 可以的，初音美美的, comment_author_id: 22498786; comment: 大大霸屏初音！😂, comment_author_id: 20663434; comment: 很美(love4), comment_author_id: 22707244; comment: 美(/≧▽≦)/~┴┴ , comment_author_id: 22690482; comment: かわい, comment_author_id: 18255423; comment: 好美 (love4), comment_author_id: 19329345; comment: 厉害了, comment_author_id: 15078056; comment: so good!, comment_author_id: 22320445; comment: 双击评论666, comment_author_id: 22236542; comment: 好看！！, comment_author_id: 22157741; comment: 好看！！！, comment_author_id: 21335032; comment: 这张シタ屌爆了, comment_author_id: 18125932; comment: あなたの投稿の中でいつも好んでいて作品をたくさん見ました、まさか全ては一人の作品だと思いませんでした！, comment_author_id: 18125932; comment: 在您的投稿看到了好多之前一直非常喜欢的图，真没想到作者都是一个人！, comment_author_id: 21628625; comment: 嗯，经典的图了(normal2), comment_author_id: 21426662; comment: 一目見た時からすっごく惹かれましたっ！初音ミクがすごく綺麗で感動しました（≧∇≦）, comment_author_id: 8731654; comment: (sweat4)上次看才刚破30w怎么两个月就四十万了。。。。
Image Mode = bigNew
Pages      = 1
Date       = 2013年2月20日 00:20
Resolution = 1400x986
Tools      = SAI
BookmarkCount= 55617
Link       = http://www.pixiv.net/member_illust.php?mode=medium&illust_id=33703665
Ugoira Data= """


class TestDataUtilMethods(unittest.TestCase):
    def test_parse_comments(self):
        input_comments = "comment_author_id: 22690482; comment: かわい, comment_author_id: 18255423; comment: 好美 (love4)"
        actual_output = parse_comments(input_comments)
        expected_output = [Comment(22690482, "かわい"), Comment(18255423, "好美 (love4)")]
        self.assertItemsEqual(actual_output, expected_output)

    def test_parse_image_name_info(self):
        input_file_name = '33703665_p0 - glow.jpg'
        actual_output = parse_image_name_info(input_file_name)
        expected_output = (33703665, 0, "glow", "jpg")
        self.assertEqual(actual_output, expected_output)

    def test_parse_pixiv_info(self):
        dirpath = tempfile.mkdtemp()
        image_path = dirpath + '/33703665_p0 - glow.jpg'
        txt_path = dirpath + '/33703665_p0 - glow.jpg.txt'
        with open(txt_path, 'w') as f:
            f.write(PIXIV_INFO_CONTENT.encode('utf8'))
        actual_pixiv_info = parse_pixiv_info(image_path)
        expected_pixiv_info = PixivInfo(artist_id=163536,
                                        artist_name="Rella",
                                        image_id=33703665,
                                        title="glow",
                                        caption="きっと このまま君を 溶かして夜になるだけ",
                                        tags="初音ミク, 素晴らしい, 夕焼け, ふつくしい, クリック推奨, SAIの本気, VOCALOID, メイキング希望!, これはいい初音, VOCALOID50000users入り".split(
                                            ', '),
                                        comments=parse_comments(
                                            "comment_author_id: 22917715; comment: 可以的，初音美美的, comment_author_id: 22498786; comment: 大大霸屏初音！😂, comment_author_id: 20663434; comment: 很美(love4), comment_author_id: 22707244; comment: 美(/≧▽≦)/~┴┴ , comment_author_id: 22690482; comment: かわい, comment_author_id: 18255423; comment: 好美 (love4), comment_author_id: 19329345; comment: 厉害了, comment_author_id: 15078056; comment: so good!, comment_author_id: 22320445; comment: 双击评论666, comment_author_id: 22236542; comment: 好看！！, comment_author_id: 22157741; comment: 好看！！！, comment_author_id: 21335032; comment: 这张シタ屌爆了, comment_author_id: 18125932; comment: あなたの投稿の中でいつも好んでいて作品をたくさん見ました、まさか全ては一人の作品だと思いませんでした！, comment_author_id: 18125932; comment: 在您的投稿看到了好多之前一直非常喜欢的图，真没想到作者都是一个人！, comment_author_id: 21628625; comment: 嗯，经典的图了(normal2), comment_author_id: 21426662; comment: 一目見た時からすっごく惹かれましたっ！初音ミクがすごく綺麗で感動しました（≧∇≦）, comment_author_id: 8731654; comment: (sweat4)上次看才刚破30w怎么两个月就四十万了。。。。"),
                                        image_mode="bigNew",
                                        pages=1,
                                        date=datetime.datetime(2013, 2, 20, 0, 20),
                                        resolution="1400x986",
                                        tools="SAI",
                                        bookmark_count=55617,
                                        link="http://www.pixiv.net/member_illust.php?mode=medium&illust_id=33703665",
                                        ugoira_data="",
                                        description_urls=[])
        shutil.rmtree(dirpath)
        self.assertEqual(actual_pixiv_info, expected_pixiv_info)

    def test_create_database(self):
        dirpath = tempfile.mkdtemp()
        image_path = dirpath + '/33703665_p0 - glow.jpg'
        with open(image_path, 'w') as f:
            pass
        txt_path = dirpath + '/33703665_p0 - glow.jpg.txt'
        with open(txt_path, 'w') as f:
            f.write(PIXIV_INFO_CONTENT.encode('utf8'))
        actual_pixiv_info = parse_pixiv_info(image_path)

        actual_output = create_database(dirpath)
        expected_output = {33703665: actual_pixiv_info}

        shutil.rmtree(dirpath)
        self.assertDictEqual(actual_output, expected_output)

    def test_create_labels(self):
        # pixiv_info_1 = PixivInfo(artist_id=163536,
        #                      artist_name="Rella",
        #                      image_id=33703665,
        #                      title="glow",
        #                      caption="きっと このまま君を 溶かして夜になるだけ",
        #                      tags="初音ミク, 素晴らしい, 夕焼け, ふつくしい, クリック推奨, SAIの本気, VOCALOID, メイキング希望!, これはいい初音, VOCALOID50000users入り".split(', '),
        #                      comments=parse_comments("comment_author_id: 22917715; comment: 可以的，初音美美的, comment_author_id: 22498786; comment: 大大霸屏初音！😂, comment_author_id: 20663434; comment: 很美(love4), comment_author_id: 22707244; comment: 美(/≧▽≦)/~┴┴ , comment_author_id: 22690482; comment: かわい, comment_author_id: 18255423; comment: 好美 (love4), comment_author_id: 19329345; comment: 厉害了, comment_author_id: 15078056; comment: so good!, comment_author_id: 22320445; comment: 双击评论666, comment_author_id: 22236542; comment: 好看！！, comment_author_id: 22157741; comment: 好看！！！, comment_author_id: 21335032; comment: 这张シタ屌爆了, comment_author_id: 18125932; comment: あなたの投稿の中でいつも好んでいて作品をたくさん見ました、まさか全ては一人の作品だと思いませんでした！, comment_author_id: 18125932; comment: 在您的投稿看到了好多之前一直非常喜欢的图，真没想到作者都是一个人！, comment_author_id: 21628625; comment: 嗯，经典的图了(normal2), comment_author_id: 21426662; comment: 一目見た時からすっごく惹かれましたっ！初音ミクがすごく綺麗で感動しました（≧∇≦）, comment_author_id: 8731654; comment: (sweat4)上次看才刚破30w怎么两个月就四十万了。。。。"),
        #                      image_mode="bigNew",
        #                      pages=1,
        #                      date=datetime.datetime(2013, 2, 20, 0, 20),
        #                      resolution="1400x986",
        #                      tools="SAI",
        #                      bookmark_count=55617,
        #                      link="http://www.pixiv.net/member_illust.php?mode=medium&illust_id=33703665",
        #                      ugoira_data="")
        # pixiv_info_2 = PixivInfo(artist_id=163536,
        #                      artist_name="Rella",
        #                      image_id=33703665,
        #                      title="glow",
        #                      caption="きっと このまま君を 溶かして夜になるだけ",
        #                      tags="初音ミク, 素晴らしい, 夕焼け, ふつくしい, クリック推奨, SAIの本気, VOCALOID, メイキング希望!, これはいい初音, VOCALOID50000users入り".split(', '),
        #                      comments=parse_comments("comment_author_id: 22917715; comment: 可以的，初音美美的, comment_author_id: 22498786; comment: 大大霸屏初音！😂, comment_author_id: 20663434; comment: 很美(love4), comment_author_id: 22707244; comment: 美(/≧▽≦)/~┴┴ , comment_author_id: 22690482; comment: かわい, comment_author_id: 18255423; comment: 好美 (love4), comment_author_id: 19329345; comment: 厉害了, comment_author_id: 15078056; comment: so good!, comment_author_id: 22320445; comment: 双击评论666, comment_author_id: 22236542; comment: 好看！！, comment_author_id: 22157741; comment: 好看！！！, comment_author_id: 21335032; comment: 这张シタ屌爆了, comment_author_id: 18125932; comment: あなたの投稿の中でいつも好んでいて作品をたくさん見ました、まさか全ては一人の作品だと思いませんでした！, comment_author_id: 18125932; comment: 在您的投稿看到了好多之前一直非常喜欢的图，真没想到作者都是一个人！, comment_author_id: 21628625; comment: 嗯，经典的图了(normal2), comment_author_id: 21426662; comment: 一目見た時からすっごく惹かれましたっ！初音ミクがすごく綺麗で感動しました（≧∇≦）, comment_author_id: 8731654; comment: (sweat4)上次看才刚破30w怎么两个月就四十万了。。。。"),
        #                      image_mode="bigNew",
        #                      pages=1,
        #                      date=datetime.datetime(2013, 2, 20, 0, 20),
        #                      resolution="1400x986",
        #                      tools="SAI",
        #                      bookmark_count=55617,
        #                      link="http://www.pixiv.net/member_illust.php?mode=medium&illust_id=33703665",
        #                      ugoira_data="")
        # database = {:}
        # create_labels()
        raise NotImplementedError

    def test_logits_to_labels(self):
        index2tag = {0: "0", 1: "1", 2: "2", 3: "3"}
        # First test boolean input
        logits = np.array([True, False, False, True])
        actual_output = logits_to_labels(logits, index2tag)
        expected_output = ["0", "3"]
        self.assertItemsEqual(actual_output, expected_output)

        # Next test probability input
        logits = np.array([0.5, 0.1, 0.49, 0.9])
        actual_output = logits_to_labels(logits, index2tag)
        expected_output = ["0", "3"]
        self.assertItemsEqual(actual_output, expected_output)

    def test_calc_sparse_pca(self):
        n_samples = 1000
        n_features = 30
        n_components = 20
        data = sparse_random_matrix(n_samples, n_features, density=0.01, random_state=42)
        actual_output = calc_sparse_pca(data, n_components=n_components)
        self.assertEqual(actual_output[0].shape, (n_samples,))
        self.assertEqual(actual_output[1].shape, (n_components, n_features))
        print(actual_output[1])

    def test_calc_pca_time(self):
        n_samples = 10000
        n_features = 1000
        n_components = 20
        num_iter = 10
        data = sparse_random_matrix(n_samples, n_features, density=0.01, random_state=42)
        data = data.toarray()
        start_time = time.time()
        for _ in range(num_iter):
            actual_output = calc_sparse_pca(data, n_components=n_components)
        end_time = time.time()
        sparse_pca_elapsed_time = (end_time-start_time) / num_iter

        print("Elapsed time for calc_sparse_pca is %f" %(sparse_pca_elapsed_time))

        start_time = time.time()
        for _ in range(num_iter):
            actual_output = calc_pca(data, n_components=n_components)
        end_time = time.time()
        pca_elapsed_time = (end_time-start_time) / num_iter

        print("Elapsed time for calc_pca is %f" %(pca_elapsed_time))
        self.assertGreater(pca_elapsed_time, sparse_pca_elapsed_time)

    def test_cluster_images(self):
        input_dir='/home/xor/PycharmProjects/PixivUtil2/pixiv_downloaded/Rella (163536)'
        db_dir="sanity_check_db.pkl"
        output_dir="sanity_check_cluster_images/"
        tag_max_count=50
        num_clusters=8
        cluster_images(input_dir,db_dir,output_dir,tag_max_count, num_clusters)

    def test_get_image_paths_subdir(self):
        input_dir = 'pixiv_downloaded/'
        image_paths = ["pixiv_downloaded/1.jpg", "pixiv_downloaded/2.png"]
        actual_output = get_image_paths_subdir(image_paths, input_dir)
        expected_output = ["1.jpg", "2.png"]
        self.assertEqual(actual_output, expected_output)