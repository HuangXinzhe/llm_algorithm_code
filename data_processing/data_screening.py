"""
数据筛选
"""
from deita.pipeline import Pipeline
import argparse

# 创建一个解析器
parser = argparse.ArgumentParser()

# 添加参数
parser.add_argument("--data_path",
                    type=str,
                    required=True,
                    help="json file with sharegpt format")
parser.add_argument("--other_data_path",
                    type=str,
                    required=True,
                    help="embedding file path (pickle format)")
parser.add_argument("--threshold",
                    type=float,
                    default=0.9,
                    help="filter threshold default: 0.9")
parser.add_argument("--data_size",
                    type=int,
                    default=None,
                    help="size of selected data")
parser.add_argument("--chunk_size",
                    type=int,
                    default=100000,
                    help="used for more efficient GPU computing  default: 100000")
parser.add_argument("--sort_key",
                    type=str,
                    default="complexity_scores,quality_scores",
                    help="default: complexity_scores,quality_scores")
parser.add_argument("--output_path",
                    type=str,
                    required=True,
                    help="json format output path")
parser.add_argument("--distance_metric",
                    type=str,
                    default="cosine",
                    help="default: cosine")
parser.add_argument("--embedding_field",
                    type=str,
                    default="embedding",
                    help="default: embedding")
parser.add_argument("--is_compression",
                    type=bool,
                    default=False,
                    help="default: False")
parser.add_argument("--device",
                    type=int,
                    default=0,
                    help="GPU IDX, default: 0")


# 解析参数
args = parser.parse_args()

filter_pipeline = Pipeline("filter_pipeline",
                           data_path=args.data_path,  # json file with sharegpt format
                           # embedding file path (pickle format)
                           other_data_path=args.other_data_path,
                           threshold=args.threshold,  # filter threshold default: 0.9
                           data_size=args.data_size,  # size of selected data
                           chunk_size=args.chunk_size,  # used for more efficient GPU computing  default: 100000
                           sort_key=args.sort_key,  # default: "complexity_scores,quality_scores"
                           output_path=args.output_path,  # json format output path
                           distance_metric=args.distance_metric,  # default: cosine
                           embedding_field=args.embedding_field,  # default: embedding
                           is_compression=args.is_compression,  # default: False
                           device=args.device  # GPU IDX, default: 0
                           )

filter_pipeline.run()
