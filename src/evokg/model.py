from collections import namedtuple

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
from torch_scatter import scatter_mean

import settings
from evokg.gnn import RGCN
from evokg.tpp import LogNormMixTPP
from utils.model_utils import node_norm_to_edge_norm, get_embedding

MultiAspectEmbedding = namedtuple('MultiAspectEmbedding', ['structural', 'temporal'], defaults=[None, None])


class EmbeddingUpdater(nn.Module):
    def __init__(self, num_nodes, in_dim, structural_hid_dim, temporal_hid_dim, graph_structural_conv, graph_temporal_conv,
                 node_latest_event_time, num_rels, rel_embed_dim, num_gconv_layers=2, num_rnn_layers=1,
                 time_interval_transform=None, dropout=0.0, activation=None, graph_name=None):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.in_dim = in_dim
        self.structural_hid_dim = structural_hid_dim
        self.temporal_hid_dim = temporal_hid_dim
        self.node_latest_event_time = node_latest_event_time

        if graph_structural_conv in ['RGCN+GRU', 'RGCN+RNN']:
            gconv, rnn = graph_structural_conv.split("+")
            self.graph_structural_conv = \
                GraphStructuralRNNConv(gconv, num_gconv_layers, rnn, num_rnn_layers, in_dim, structural_hid_dim,
                                       num_nodes, num_rels, rel_embed_dim, dropout=dropout, activation=activation, graph_name=graph_name)
        elif graph_structural_conv is None:
            self.graph_structural_conv = None
        else:
            raise ValueError(f"Invalid graph structural conv: {graph_structural_conv}")

        if graph_temporal_conv in ['RGCN+GRU', 'RGCN+RNN']:
            gconv, rnn = graph_temporal_conv.split("+")
            self.graph_temporal_conv = \
                GraphTemporalRNNConv(gconv, num_gconv_layers, rnn, num_rnn_layers, in_dim, temporal_hid_dim,
                                     node_latest_event_time, time_interval_transform, num_nodes, num_rels,
                                     dropout=dropout, activation=activation, graph_name=graph_name)
        elif graph_temporal_conv is None:
            self.graph_temporal_conv = None
        else:
            raise ValueError(f"Invalid graph temporal conv: {graph_temporal_conv}")

        self.structural_relation_rnn = RelationRNN("RNN", num_rnn_layers, in_dim, rel_embed_dim, num_rels, dropout)
        self.temporal_relation_rnn = RelationRNN("RNN", num_rnn_layers, in_dim, rel_embed_dim, num_rels, dropout)

    def forward(self, prior_G, batch_G, cumul_G, static_entity_emb, dynamic_entity_emb,
                dynamic_relation_emb, device, batch_node_indices=None):
        assert all([emb.device == torch.device('cpu') for emb in dynamic_entity_emb]), [emb.device for emb in dynamic_entity_emb]

        batch_G = batch_G.to(device)
        if batch_node_indices is None:
            batch_node_indices = batch_G.nodes().long()  # update embeddings of all nodes in batch_G

        if self.graph_structural_conv is None:
            batch_structural_dynamic_entity_emb = None
        else:
            batch_structural_dynamic_entity_emb = \
                self.graph_structural_conv(batch_G, dynamic_entity_emb, static_entity_emb, device, batch_node_indices)
        if self.graph_temporal_conv is None:
            batch_temporal_dynamic_entity_emb = None
        else:
            batch_temporal_dynamic_entity_emb = \
                self.graph_temporal_conv(batch_G, dynamic_entity_emb, static_entity_emb, device, batch_node_indices)

        batch_structural_dynamic_relation_emb = \
            self.structural_relation_rnn.forward(batch_G, dynamic_relation_emb.structural, static_entity_emb.structural, device)
        batch_temporal_dynamic_relation_emb = \
            self.temporal_relation_rnn.forward(batch_G, dynamic_relation_emb.temporal, static_entity_emb.temporal, device)

        """Update dynamic entity emb"""
        updated_structural = dynamic_entity_emb.structural
        if batch_structural_dynamic_entity_emb is not None:
            updated_structural = dynamic_entity_emb.structural.clone()
            updated_structural[batch_G.ndata[dgl.NID][batch_node_indices].long()] = batch_structural_dynamic_entity_emb.cpu()
        updated_temporal = dynamic_entity_emb.temporal
        if batch_temporal_dynamic_entity_emb is not None:
            updated_temporal = dynamic_entity_emb.temporal.clone()
            updated_temporal[batch_G.ndata[dgl.NID][batch_node_indices].long()] = batch_temporal_dynamic_entity_emb.cpu()
        updated_dynamic_entity_emb = MultiAspectEmbedding(structural=updated_structural, temporal=updated_temporal)

        """Update dynamic relation emb"""
        batch_G_rel = batch_G.edata['rel_type']
        batch_G_uniq_rel = torch.unique(batch_G_rel, sorted=True).long()

        updated_structural = dynamic_relation_emb.structural
        if batch_structural_dynamic_relation_emb is not None:
            updated_structural = dynamic_relation_emb.structural.clone()
            updated_structural[batch_G_uniq_rel] = batch_structural_dynamic_relation_emb.cpu()
        updated_temporal = dynamic_relation_emb.temporal
        if batch_temporal_dynamic_relation_emb is not None:
            updated_temporal = dynamic_relation_emb.temporal.clone()
            updated_temporal[batch_G_uniq_rel] = batch_temporal_dynamic_relation_emb.cpu()
        updated_dynamic_relation_emb = MultiAspectEmbedding(structural=updated_structural, temporal=updated_temporal)

        return updated_dynamic_entity_emb, updated_dynamic_relation_emb


class GraphStructuralRNNConv(nn.Module):
    def __init__(self, graph_conv, num_gconv_layers, rnn, num_rnn_layers, in_dim, hid_dim, num_nodes, num_rels, rel_embed_dim,
                 add_entity_emb=False, dropout=0.0, activation=None, graph_name=None):
        super().__init__()

        self.num_nodes = num_nodes  # num nodes in the entire G
        self.num_rels = num_rels

        if 'RGCN' == graph_conv:
            self.graph_conv = RGCN(in_dim, hid_dim, hid_dim, n_layers=num_gconv_layers,
                                   num_rels=self.num_rels, regularizer="bdd",
                                   num_bases=50 if graph_name == "GDELT" else 100, dropout=dropout,
                                   activation=activation, layer_norm=False)
        else:
            raise ValueError(f"Invalid graph conv: {graph_conv}")

        structural_rnn_in_dim = hid_dim
        self.add_entity_emb = add_entity_emb
        if self.add_entity_emb:
            structural_rnn_in_dim += hid_dim

        if rnn == "GRU":
            self.rnn_structural = nn.GRU(input_size=structural_rnn_in_dim, hidden_size=hid_dim,
                                         num_layers=num_rnn_layers, batch_first=True, dropout=0.0)
        elif rnn == "RNN":
            self.rnn_structural = nn.RNN(input_size=structural_rnn_in_dim, hidden_size=hid_dim,
                                         num_layers=num_rnn_layers, batch_first=True, dropout=0.0)
        else:
            raise ValueError(f"Invalid rnn: {rnn}")

        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_G, dynamic_entity_emb, static_entity_emb, device, batch_node_indices=None):
        if batch_node_indices is None:
            batch_node_indices = batch_G.nodes().long()  # update embeddings of all nodes in batch_G

        """Structural RNN input"""
        batch_structural_static_entity_emb = static_entity_emb.structural[batch_G.ndata[dgl.NID].long()].to(device)
        if isinstance(self.graph_conv, RGCN):
            edge_norm = node_norm_to_edge_norm(batch_G)
            conv_structural_static_emb = self.graph_conv(batch_G, batch_structural_static_entity_emb, batch_G.edata['rel_type'].long(), edge_norm)
        else:
            conv_structural_static_emb = self.graph_conv(batch_G, batch_structural_static_entity_emb)  # shape=(# nodes in batch_G, dim-hidden)

        structural_rnn_input = [
            conv_structural_static_emb[batch_node_indices],
        ]
        if self.add_entity_emb:
            structural_rnn_input.append(static_entity_emb.structural[batch_G.ndata[dgl.NID][batch_node_indices].long()].to(device))
        structural_rnn_input = torch.cat(structural_rnn_input, dim=1).unsqueeze(1)

        # Update structural dynamics
        structural_dynamic = dynamic_entity_emb.structural[batch_G.ndata[dgl.NID][batch_node_indices].long()]
        structural_dynamic = structural_dynamic.to(device)

        output, hn = self.rnn_structural(structural_rnn_input, structural_dynamic.transpose(0, 1).contiguous())  # transpose to make shape to be (num_layers, batch, hidden_size)
        updated_structural_dynamic_entity_emb = hn.transpose(0, 1)  # transpose to make shape to be (batch, num_layers, hidden_size)

        return updated_structural_dynamic_entity_emb

    def extra_repr(self):
        field_desc = [f"add_entity_emb={self.add_entity_emb}"]
        return ", ".join(field_desc)


class GraphTemporalRNNConv(nn.Module):
    def __init__(self, graph_conv, num_gconv_layers, rnn, num_rnn_layers, in_dim, hid_dim,
                 node_latest_event_time, time_interval_transform, num_nodes, num_rels,
                 dropout=0.0, activation=None, graph_name=None):
        super().__init__()

        self.num_nodes = num_nodes  # num nodes in the entire G
        self.num_rels = num_rels

        if 'RGCN' == graph_conv:
            self.graph_conv = RGCN(in_dim, hid_dim, hid_dim, n_layers=num_gconv_layers,
                                   num_rels=self.num_rels, regularizer="bdd",
                                   num_bases=50 if graph_name == "GDELT" else 100, dropout=dropout,
                                   activation=activation, layer_norm=False)
        else:
            raise ValueError(f"Invalid graph conv: {graph_conv}")

        self.node_latest_event_time = node_latest_event_time
        self.time_interval_transform = time_interval_transform

        temporal_rnn_in_dim = hid_dim
        if rnn == "GRU":
            self.rnn_temporal = nn.GRU(input_size=temporal_rnn_in_dim, hidden_size=hid_dim,
                                       num_layers=num_rnn_layers, batch_first=True, dropout=0.0)
        elif rnn == "RNN":
            self.rnn_temporal = nn.RNN(input_size=temporal_rnn_in_dim, hidden_size=hid_dim,
                                       num_layers=num_rnn_layers, batch_first=True, dropout=0.0)
        else:
            raise ValueError(f"Invalid rnn: {rnn}")

        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_G, dynamic_entity_emb, static_entity_emb, device, batch_node_indices=None):
        if batch_node_indices is None:
            batch_node_indices = batch_G.nodes().long()  # update embeddings of all nodes in batch_G

        """Inter event times (in both directions)"""
        batch_G_sparse_inter_event_times = \
            EventTimeHelper.get_sparse_inter_event_times(batch_G, self.node_latest_event_time[..., 0])
        EventTimeHelper.get_inter_event_times(batch_G, self.node_latest_event_time[..., 0], update_latest_event_time=True)

        rev_batch_G = dgl.reverse(batch_G, copy_ndata=True, copy_edata=True)
        rev_batch_G.num_relations = batch_G.num_relations
        rev_batch_G.num_all_nodes = batch_G.num_all_nodes
        rev_batch_G_sparse_inter_event_times = \
            EventTimeHelper.get_sparse_inter_event_times(rev_batch_G, self.node_latest_event_time[..., 1])
        EventTimeHelper.get_inter_event_times(rev_batch_G, self.node_latest_event_time[..., 1], update_latest_event_time=True)

        """Temporal RNN input"""
        batch_temporal_static_entity_emb = static_entity_emb.temporal[batch_G.ndata[dgl.NID].long()].to(device)
        edge_norm = (1 / self.time_interval_transform(batch_G_sparse_inter_event_times).clamp(min=1e-10)).clamp(max=10.0)
        batch_G_conv_temporal_static_emb = self.graph_conv(batch_G, batch_temporal_static_entity_emb, batch_G.edata['rel_type'].long(), edge_norm)
        temporal_rnn_input_batch_G = torch.cat([
            batch_G_conv_temporal_static_emb,
        ], dim=1)[batch_node_indices].unsqueeze(1)

        rev_batch_temporal_static_entity_emb = static_entity_emb.temporal[rev_batch_G.ndata[dgl.NID].long()].to(device)
        rev_edge_norm = (1 / self.time_interval_transform(rev_batch_G_sparse_inter_event_times).clamp(min=1e-10)).clamp(max=10.0)
        rev_batch_G_conv_temporal_static_emb = self.graph_conv(rev_batch_G, rev_batch_temporal_static_entity_emb, batch_G.edata['rel_type'].long(), rev_edge_norm)
        temporal_rnn_input_rev_batch_G = torch.cat([
            rev_batch_G_conv_temporal_static_emb,
        ], dim=1)[batch_node_indices].unsqueeze(1)

        temporal_dynamic = dynamic_entity_emb.temporal[batch_G.ndata[dgl.NID][batch_node_indices].long()].to(device)
        temporal_dynamic_batch_G = temporal_dynamic[..., 0]  # dynamics as a recipient
        temporal_dynamic_rev_batch_G = temporal_dynamic[..., 1]  # dynamics as a sender

        output, hn = self.rnn_temporal(temporal_rnn_input_batch_G, temporal_dynamic_batch_G.transpose(0, 1).contiguous())  # transpose to make shape to be (num_layers, batch, hidden_size)
        updated_temporal_dynamic_batch_G = hn.transpose(0, 1)  # transpose to make shape to be (batch, num_layers, hidden_size)

        output, hn = self.rnn_temporal(temporal_rnn_input_rev_batch_G, temporal_dynamic_rev_batch_G.transpose(0, 1).contiguous())  # transpose to make shape to be (num_layers, batch, hidden_size)
        updated_temporal_dynamic_rev_batch_G = hn.transpose(0, 1)  # transpose to make shape to be (batch, num_layers, hidden_size)

        updated_temporal_dynamic_entity_emb = torch.cat([updated_temporal_dynamic_batch_G.unsqueeze(-1),
                                                         updated_temporal_dynamic_rev_batch_G.unsqueeze(-1)], dim=-1)
        return updated_temporal_dynamic_entity_emb


class RelationRNN(nn.Module):
    def __init__(self, rnn, num_rnn_layers, rnn_in_dim, rnn_hid_dim, num_rels, dropout=0.0):
        super().__init__()

        if rnn == "GRU":
            self.rnn_relation = nn.GRU(input_size=rnn_in_dim, hidden_size=rnn_hid_dim,
                                       num_layers=num_rnn_layers, batch_first=True, dropout=dropout)
        elif rnn == "RNN":
            self.rnn_relation = nn.RNN(input_size=rnn_in_dim, hidden_size=rnn_hid_dim,
                                       num_layers=num_rnn_layers, batch_first=True, dropout=dropout)
        else:
            raise ValueError(f"Invalid rnn: {rnn}")

    def forward(self, batch_G, dynamic_relation_emb, static_entity_emb, device):
        batch_G_src, batch_G_dst = batch_G.edges()
        batch_G_rel = batch_G.edata['rel_type'].long()

        batch_G_src_nid = batch_G.ndata[dgl.NID][batch_G_src.long()].long()
        batch_G_dst_nid = batch_G.ndata[dgl.NID][batch_G_dst.long()].long()

        # aggregate entity embeddings by relation. transpose() is necessary to aggregate entity emb matrix row-wise.
        batch_G_src_emb_avg_by_rel_ = \
            scatter_mean(static_entity_emb[batch_G_src_nid].to(device).transpose(0, 1),
                         batch_G_rel).transpose(0, 1)  # shape=(max rel in batch_G, static entity emb dim)
        batch_G_dst_emb_avg_by_rel_ = \
            scatter_mean(static_entity_emb[batch_G_dst_nid].to(device).transpose(0, 1),
                         batch_G_rel).transpose(0, 1)  # shape=(max rel in batch_G, static entity emb dim)

        # filter out relations that are non-existent in batch_G
        batch_G_uniq_rel = torch.unique(batch_G_rel, sorted=True)
        batch_G_src_emb_avg_by_rel = batch_G_src_emb_avg_by_rel_[batch_G_uniq_rel]  # shape=(# uniq rels in batch_G, static entity emb dim)
        batch_G_dst_emb_avg_by_rel = batch_G_dst_emb_avg_by_rel_[batch_G_uniq_rel]  # shape=(# uniq rels in batch_G, static entity emb dim)

        batch_G_dynamic_relation_emb = dynamic_relation_emb[batch_G_uniq_rel]
        batch_G_src_dynamic_relation_emb = batch_G_dynamic_relation_emb[..., 0].to(device)
        batch_G_dst_dynamic_relation_emb = batch_G_dynamic_relation_emb[..., 1].to(device)

        output, hn = self.rnn_relation(batch_G_src_emb_avg_by_rel.unsqueeze(1),
                                       batch_G_src_dynamic_relation_emb.transpose(0, 1).contiguous())  # transpose to make shape to be (num_layers, batch, hidden_size)
        updated_batch_G_src_dynamic_relation_emb = hn.transpose(0, 1)  # transpose to make shape to be (batch, num_layers, hidden_size)
        output, hn = self.rnn_relation(batch_G_dst_emb_avg_by_rel.unsqueeze(1),
                                       batch_G_dst_dynamic_relation_emb.transpose(0, 1).contiguous())  # transpose to make shape to be (num_layers, batch, hidden_size)
        updated_batch_G_dst_dynamic_relation_emb = hn.transpose(0, 1)  # transpose to make shape to be (batch, num_layers, hidden_size)
        updated_batch_G_dynamic_relation_emb = torch.cat([updated_batch_G_src_dynamic_relation_emb.unsqueeze(-1),
                                                          updated_batch_G_dst_dynamic_relation_emb.unsqueeze(-1)], dim=-1)

        return updated_batch_G_dynamic_relation_emb


class EventTimeHelper:
    @classmethod
    def get_sparse_inter_event_times(cls, batch_G, node_latest_event_time, _global=False):
        batch_sparse_latest_event_times = cls.get_sparse_latest_event_times(batch_G, node_latest_event_time, _global)
        return batch_G.edata['time'] - batch_sparse_latest_event_times

    @classmethod
    def get_sparse_latest_event_times(cls, batch_G, node_latest_event_time, _global=False):
        batch_G_nid = batch_G.ndata[dgl.NID].long()
        batch_latest_event_time = node_latest_event_time[batch_G_nid]

        batch_G_src, batch_G_dst = batch_G.edges()
        device = batch_G.ndata[dgl.NID].device
        if _global:
            return batch_latest_event_time[batch_G_dst.long(), -1].to(device)
        else:
            return batch_latest_event_time[batch_G_dst.long(), batch_G_nid[batch_G_src.long()]].to(device)

    @classmethod
    def get_inter_event_times(cls, batch_G, node_latest_event_time, update_latest_event_time=True):
        batch_G_nid = batch_G.ndata[dgl.NID].long()
        batch_latest_event_time = node_latest_event_time[batch_G_nid]

        batch_G_src, batch_G_dst = batch_G.edges()
        batch_G_src, batch_G_dst = batch_G_src.long(), batch_G_dst.long()
        batch_G_time, batch_G_rel = batch_G.edata['time'], batch_G.edata['rel_type'].long()
        batch_G_time = batch_G_time.to(settings.INTER_EVENT_TIME_DTYPE)

        device = batch_G.ndata[dgl.NID].device
        batch_inter_event_times = torch.zeros(batch_G.num_nodes(), batch_G.num_all_nodes + 1, dtype=settings.INTER_EVENT_TIME_DTYPE).to(device)
        batch_inter_event_times[batch_G_dst, batch_G_nid[batch_G_src]] = \
            batch_G_time - batch_latest_event_time[batch_G_dst, batch_G_nid[batch_G_src]].to(device)

        batch_G.update_all(fn.copy_e('time', 't'), fn.max('t', 'max_event_time'))
        batch_G_max_event_time = batch_G.ndata['max_event_time'].to(settings.INTER_EVENT_TIME_DTYPE)

        batch_max_latest_event_time = batch_latest_event_time[:, -1].to(device)
        batch_G_max_event_time = torch.max(batch_G_max_event_time, batch_max_latest_event_time)
        batch_inter_event_times[:, -1] = batch_G_max_event_time - batch_max_latest_event_time

        if update_latest_event_time:
            node_latest_event_time[batch_G_nid[batch_G_dst], batch_G_nid[batch_G_src]] = batch_G_time.cpu()
            node_latest_event_time[batch_G_nid, -1] = batch_G_max_event_time.cpu()

        return batch_inter_event_times


class Combiner(nn.Module):
    def __init__(self, static_emb_dim, dynamic_emb_dim, static_dynamic_combine_mode,
                 graph_conv, num_rels=None, dropout=0.0, activation=None, num_gconv_layers=1):
        super().__init__()
        self.static_emb_dim = static_emb_dim
        self.dynamic_emb_dim = dynamic_emb_dim
        self.static_dynamic_combiner = StaticDynamicCombiner(static_dynamic_combine_mode, static_emb_dim, dynamic_emb_dim)

        self.num_gconv_layer = num_gconv_layers
        if graph_conv == RGCN.__name__:
            self.graph_conv_static = RGCN(self.static_emb_dim, self.static_emb_dim, self.static_emb_dim,
                                          n_layers=num_gconv_layers, num_rels=num_rels, regularizer="bdd",
                                          num_bases=100, dropout=dropout, activation=activation)
            self.graph_conv_dynamic = RGCN(self.dynamic_emb_dim, self.dynamic_emb_dim, self.dynamic_emb_dim,
                                           n_layers=num_gconv_layers, num_rels=num_rels, regularizer="bdd",
                                           num_bases=100, dropout=dropout, activation=activation)
        elif graph_conv is None:
            self.graph_conv_static = None
            self.graph_conv_dynamic = None
        else:
            raise ValueError(f"Invalid graph conv: {graph_conv}")

        self.dropout = nn.Dropout(dropout)

    @property
    def combined_emb_dim(self):
        return self.static_dynamic_combiner.combined_emb_dim

    @classmethod
    def do_graph_conv(cls, G, emb, graph_conv):
        if graph_conv is None:
            return emb

        if isinstance(graph_conv, RGCN):
            edge_norm = node_norm_to_edge_norm(G)
            return graph_conv(G, emb, G.edata['rel_type'], edge_norm)
        else:
            return graph_conv(G, emb)

    def forward(self, static_emb, dynamic_emb, G=None):
        if self.static_dynamic_combiner.use_static_emb:
            static_emb = self.do_graph_conv(G, static_emb, self.graph_conv_static)
        if self.static_dynamic_combiner.use_dynamic_emb:
            dynamic_emb = self.do_graph_conv(G, dynamic_emb, self.graph_conv_dynamic)

        combined_emb = self.static_dynamic_combiner(self.dropout(static_emb), dynamic_emb)
        return combined_emb


class StaticDynamicCombiner(nn.Module):
    def __init__(self, mode, static_emb_dim, dynamic_emb_dim):
        super().__init__()
        self.mode = mode
        self.static_emb_dim = static_emb_dim
        self.dynamic_emb_dim = dynamic_emb_dim

        if self.mode == "concat":
            self.combined_emb_dim = static_emb_dim + dynamic_emb_dim
            self.use_static_emb = True
            self.use_dynamic_emb = True
        elif self.mode == "static_only":
            self.combined_emb_dim = static_emb_dim
            self.use_static_emb = True
            self.use_dynamic_emb = False
        elif self.mode == "dynamic_only":
            self.combined_emb_dim = dynamic_emb_dim
            self.use_static_emb = False
            self.use_dynamic_emb = True
        else:
            raise ValueError(f"Invalid combiner mode: {mode}")

    def forward(self, static_emb, dynamic_emb):
        if self.mode == "concat":
            return torch.cat([static_emb, dynamic_emb], dim=1)
        elif self.mode == "static_only":
            return static_emb
        elif self.mode == "dynamic_only":
            return dynamic_emb

    def __repr__(self):
        return "%s(mode=%s, static_emb_dim=%d, dynamic_emb_dim=%d, combined_emb_dim=%d)" % \
               (self.__class__.__name__, self.mode, self.static_emb_dim, self.dynamic_emb_dim, self.combined_emb_dim)


class GraphReadout(nn.Module):
    def __init__(self, combiner: Combiner, readout_op='max', readout_node_type="static"):
        super().__init__()

        self.combiner = combiner
        self.readout_node_type = readout_node_type
        if readout_node_type == "combined":
            self.node_emb_dim = self.combiner.combined_emb_dim
        elif readout_node_type == "static":
            self.node_emb_dim = self.combiner.static_emb_dim
        elif readout_node_type == "dynamic":
            self.node_emb_dim = self.combiner.dynamic_emb_dim
        else:
            raise ValueError(f"Invalid type: {readout_node_type}")

        self.readout_op = readout_op
        if readout_op in ['max', 'min', 'mean']:
            self.graph_emb_dim = self.node_emb_dim
        elif readout_op == 'weighted_sum':
            self.graph_emb_dim = 2 * self.node_emb_dim
            self.node_gating = nn.Sequential(
                nn.Linear(self.node_emb_dim, 1),
                nn.Sigmoid()
            )
            self.node_to_graph = nn.Linear(self.node_emb_dim, self.graph_emb_dim)
        else:
            raise ValueError(f"Invalid readout: {readout_op}")

    def forward(self, G, combined, static, dynamic):
        with G.local_scope():
            emb_dict = {"combined": combined, "static": static, "dynamic": dynamic}
            node_emb_name, node_emb = emb_dict[self.readout_node_type]

            if self.readout_op in ['max', 'min', 'mean']:
                if node_emb_name not in G.ndata:
                    G.ndata[node_emb_name] = node_emb
                return dgl.readout_nodes(G, node_emb_name, op=self.readout_op)
            elif self.readout_op == 'weighted_sum':
                return (self.node_gating(node_emb) * self.node_to_graph(node_emb)).sum(0, keepdim=True)
            else:
                raise ValueError(f"Invalid readout: {self.readout_op}")


class EdgeModel(nn.Module):
    def __init__(self, num_entities, num_rels, rel_embed_dim, combiner, dropout=0.0, graph_readout_op='max'):
        super().__init__()

        self.num_entities = num_entities
        self.num_rels = num_rels
        assert isinstance(rel_embed_dim, int)
        self.rel_embed_dim = rel_embed_dim
        self.rel_embeds = get_embedding(num_rels, rel_embed_dim)
        self.combiner = combiner
        self.combined_emb_dim = combiner.combined_emb_dim
        self.graph_readout = GraphReadout(self.combiner, graph_readout_op)

        graph_emb_dim = self.graph_readout.graph_emb_dim
        self.transform_head = nn.Sequential(
            nn.Linear(graph_emb_dim, 4 * graph_emb_dim),
            nn.Tanh(),
            nn.Linear(4 * graph_emb_dim, num_entities)
        )

        node_graph_emb_dim = self.combined_emb_dim + self.graph_readout.graph_emb_dim
        self.transform_rel = nn.Sequential(
            nn.Linear(node_graph_emb_dim, node_graph_emb_dim),
            nn.Tanh(),
            nn.Linear(node_graph_emb_dim, self.num_rels)
        )

        node_graph_rel_emb_dim = self.combined_emb_dim + self.graph_readout.graph_emb_dim + rel_embed_dim * 2
        self.transform_tail = nn.Sequential(
            nn.Linear(node_graph_rel_emb_dim, 2 * node_graph_rel_emb_dim),
            nn.Tanh(),
            nn.Linear(2 * node_graph_rel_emb_dim, num_entities)
        )

        self.dropout = nn.Dropout(dropout)
        self.criterion = nn.CrossEntropyLoss()

    def log_prob_head(self, graph_emb, G, edge_head):
        emb = graph_emb.repeat(len(edge_head), 1)
        emb = self.dropout(emb)
        head_pred = self.transform_head(emb)

        return - self.criterion(head_pred, G.ndata[dgl.NID][edge_head.long()].long()), head_pred

    def log_prob_rel(self, edge_head_emb, graph_emb, edge_rels):
        graph_emb_repeat = graph_emb.repeat(len(edge_head_emb), 1)

        emb = torch.cat((edge_head_emb, graph_emb_repeat), dim=1)
        emb = self.dropout(emb)
        rel_pred = self.transform_rel(emb)

        return - self.criterion(rel_pred, edge_rels.long()), rel_pred

    def log_prob_tail(self, edge_head_emb, graph_emb, edge_rels, G, edge_tail, dynamic_relation_emb=None):
        graph_emb_repeat = graph_emb.repeat(len(edge_head_emb), 1)

        edge_static_rel_embeds = self.rel_embeds[edge_rels.long()]
        edge_dynamic_rel_embeds = dynamic_relation_emb[edge_rels.long()]
        edge_rel_embeds = torch.cat((edge_static_rel_embeds, edge_dynamic_rel_embeds), dim=1)

        emb = torch.cat((edge_head_emb, graph_emb_repeat, edge_rel_embeds), dim=1)
        emb = self.dropout(emb)
        tail_pred = self.transform_tail(emb)

        return - self.criterion(tail_pred, G.ndata[dgl.NID][edge_tail.long()].long()), tail_pred

    def graph_emb(self, G, combined_emb, static_emb, dynamic_emb):
        return self.graph_readout.forward(G, ('emb', combined_emb), ('static_emb', static_emb), ('dynamic_emb', dynamic_emb))

    def forward(self, G, combined_emb, static_emb, dynamic_emb, dynamic_relation_emb, eid=None, return_pred=False):
        with G.local_scope():
            G.ndata['emb'] = combined_emb

            edge_head, edge_tail = G.edges()
            edge_rel = G.edata['rel_type']
            if eid is not None:
                edge_head, edge_tail, edge_rel = edge_head[eid], edge_tail[eid], edge_rel[eid]

            edge_head_emb = G.ndata['emb'][edge_head.long()]  # [# edges, emb-dim]
            assert len(edge_head_emb.size()) == 2, edge_head_emb.size()

            dynamic_rel_emb = dynamic_relation_emb[:, :, 1]  # use (relation-dest) context
            graph_emb = self.graph_emb(G, combined_emb, static_emb, dynamic_emb)
            log_prob_tail, tail_pred = self.log_prob_tail(edge_head_emb, graph_emb, edge_rel, G, edge_tail,
                                                          dynamic_rel_emb)
            log_prob_rel, rel_pred = self.log_prob_rel(edge_head_emb, graph_emb, edge_rel)
            log_prob_head, head_pred = self.log_prob_head(graph_emb, G, edge_head)
            log_prob = log_prob_tail + 0.2 * log_prob_rel + 0.1 * log_prob_head

            if return_pred:
                return log_prob, head_pred, rel_pred, tail_pred
            else:
                return log_prob


class InterEventTimeModel(nn.Module):
    def __init__(self,
                 dynamic_entity_embed_dim,
                 static_entity_embed_dim,
                 num_rels,
                 rel_embed_dim,
                 num_mix_components,
                 time_interval_transform,
                 inter_event_time_mode,
                 mean_log_inter_event_time=0.0,
                 std_log_inter_event_time=1.0,
                 dropout=0.0):
        super().__init__()
        self.tpp_model = LogNormMixTPP(dynamic_entity_embed_dim, static_entity_embed_dim, num_rels, rel_embed_dim,
                                       inter_event_time_mode, num_mix_components, time_interval_transform,
                                       mean_log_inter_event_time, std_log_inter_event_time, dropout)

    def log_prob_density(self, batch_G, dynamic_entity_emb, static_entity_emb, dynamic_relation_emb,
                         node_latest_event_time, batch_eid=None, reduction=None):
        return self.tpp_model.log_prob_density(batch_G, dynamic_entity_emb, static_entity_emb, dynamic_relation_emb,
                                               node_latest_event_time, batch_eid, reduction)

    def log_prob_interval(self, batch_G, dynamic_entity_emb, static_entity_emb, dynamic_relation_emb,
                          node_latest_event_time, batch_eid=None, reduction=None):
        return self.tpp_model.log_prob_interval(batch_G, dynamic_entity_emb, static_entity_emb, dynamic_relation_emb,
                                                node_latest_event_time, batch_eid, reduction)

    def expected_event_time(self, batch_G, dynamic_entity_emb, static_entity_emb, dynamic_relation_emb,
                            node_latest_event_time, batch_eid=None):
        return self.tpp_model.expected_event_time(batch_G, dynamic_entity_emb, static_entity_emb, dynamic_relation_emb,
                                                  node_latest_event_time, batch_eid)


class Model(nn.Module):
    def __init__(self, embedding_updater: EmbeddingUpdater, combiner, edge_model, inter_event_time_model, node_latest_event_time):
        super().__init__()
        self.embedding_updater = embedding_updater
        self.combiner = combiner
        self.edge_model = edge_model
        self.inter_event_time_model = inter_event_time_model

        self.node_latest_event_time = node_latest_event_time
