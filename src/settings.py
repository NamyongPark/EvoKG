import os
import torch
settings_path = os.path.dirname(__file__)

DATA_ROOT = os.path.join(settings_path, os.pardir, 'data')
CACHE_ROOT = os.path.join(settings_path, os.pardir, 'cache')
RESULT_ROOT = os.path.join(settings_path, os.pardir, 'result')

GRAPH_ICEWS18 = "ICEWS18"
GRAPH_ICEWS14 = "ICEWS14"
GRAPH_ICEWS_500 = "ICEWS_500"
GRAPH_GDELT = "GDELT"
GRAPH_WIKI = "WIKI"
GRAPH_YAGO = "YAGO"
ALL_GRAPHS = [GRAPH_ICEWS18, GRAPH_ICEWS14, GRAPH_ICEWS_500, GRAPH_GDELT, GRAPH_WIKI, GRAPH_YAGO]

DGL_GRAPH_ID_TYPE = torch.int32
INTER_EVENT_TIME_DTYPE = torch.float32
