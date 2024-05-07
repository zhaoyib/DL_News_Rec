'''
@File      :   test_MINDDataFrame.py
@Time      :   2024/04/25 16:47:05
@Author    :   YiboZhao 
@Version   :   1.0
@Site      :   https://github.com/zhaoyib
@RecentInfo:   test_MINDdataset.
'''
from const.path import MIND_SMALL_VAL_DATASET_DIR
from data_process.dataframe import read_behavior_df, read_news_df


def test_read_news_df() -> None:
    small_val_news_tsv_path = MIND_SMALL_VAL_DATASET_DIR / "news.tsv"
    news_df = read_news_df(small_val_news_tsv_path)
    assert news_df.columns == ["news_id", "category", "subcategory", "title", "abstract", "url"]
    assert len(news_df) == 42416


def test_read_news_df_with_entities() -> None:
    small_val_news_tsv_path = MIND_SMALL_VAL_DATASET_DIR / "news.tsv"
    news_df = read_news_df(small_val_news_tsv_path, has_entities=True)
    assert news_df.columns == [
        "news_id",
        "category",
        "subcategory",
        "title",
        "abstract",
        "url",
        "title_entities",
        "abstract_entities",
    ]
    assert len(news_df) == 42416


def test_read_behavior_df() -> None:
    small_val_behavior_tsv_path = MIND_SMALL_VAL_DATASET_DIR / "behaviors.tsv"
    behavior_df = read_behavior_df(small_val_behavior_tsv_path)
    assert behavior_df.columns == ["impression_id", "user_id", "time", "history", "impressions"]
    assert len(behavior_df) == 73152
    
if __name__== "__main__":
    test_read_behavior_df()
    test_read_news_df()
    test_read_news_df_with_entities()