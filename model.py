from imports import *
from FSpool import *
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

class GRU_graph(nn.Module):
    """
    The rnn which computes graph-level hidden state at each step i, 
    from X(i-1), F(i-1) and hidden state from edge level rnn.
    1 layer at input, rnn with num_layers, output hidden state
    """
    def __init__(self, max_num_node,
                 input_size, embedding_size,
                 hidden_size, num_layers,
                 bias_constant):
        super(GRU_graph, self).__init__()

        # Define the architecture
        self.max_num_node = max_num_node
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # input
        self.input = nn.Sequential(
            nn.Linear(input_size, embedding_size),
            nn.ReLU()
        )
        # rnn
        self.rnn = nn.GRU(input_size=embedding_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        # initialialization
        self.hidden = None
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, bias_constant)
            elif 'weight' in name:
                #nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('sigmoid'))
                nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('relu'))
                #nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                #m.weight.data = init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
    
    # Initial state of GRU_graph is 0.
    def init_hidden(self, batch_size, random_noise=False):
        hidden_init = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(DEVICE)
        if random_noise:
            std = 0.05
            noise_sampler = torch.distributions.MultivariateNormal(torch.zeros(self.hidden_size), std*torch.eye(self.hidden_size))
            hidden_init[0] = noise_sampler.sample((batch_size,))

        return hidden_init

    def forward(self, input_raw, input_len, pack=False):
        # input
        #print('input_raw', input_raw.shape)
        input_ggru = self.input(input_raw)
        #print('input_ggru', input_ggru.shape, input_ggru)
        if pack:
            input_packed = pack_padded_sequence(input_ggru, input_len, batch_first=True, enforce_sorted=False)
        else:
            input_packed = input_ggru
        # rnn
        output_raw, self.hidden = self.rnn(input_packed, self.hidden)
        if pack:
            output_raw, seq_len = pad_packed_sequence(output_raw, batch_first=True, padding_value=0.0, 
                                                      total_length=self.max_num_node)
        #print('output_raw', output_raw.shape, output_raw)
        return output_raw
    


class MLP_node(nn.Module):
    """
    2 layer Multilayer perceptron with sigmoid output to get node categories.
    2 layered fully connected with ReLU
    """
    def __init__(self, h_graph_size, embedding_size, node_size):
        super(MLP_node, self).__init__()

        self.output = nn.Sequential(
            nn.Linear(h_graph_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, node_size)
        )
        ## initialialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('sigmoid'))
                #m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                #m.weight.data = init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')

    def forward(self, h):
        y = self.output(h)
        return y



class MLP_node_MHP(nn.Module):
    """
    2 layer Multilayer perceptron with sigmoid output to get node categories for num_generators.
    2 layered fully connected with ReLU
    """
    def __init__(self, h_graph_size, embedding_size, node_size, num_generators):
        super(MLP_node_MHP, self).__init__()
        self.node_size = node_size
        self.num_generators = num_generators
        self.output = nn.Sequential(
            nn.Linear(h_graph_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, node_size*num_generators)
        )
        ## initialialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                #m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('sigmoid'))
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                #m.weight.data = init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')

    def forward(self, h):
        y = self.output(h)
        y = y.view((y.shape[0], y.shape[1], self.node_size, self.num_generators))
        return y

    
# class GRU_edge(nn.Module):
#     """
#     Sequential NN which outputs the edge categories F(i) using GRU_graph hidden state and X(i)
#     1 layer at input, rnn with hidden layers, 2 layer at output
#     """
#     def __init__(self,
#                  input_size, embedding_size,
#                  h_edge_size, num_layers,
#                  emb_edge_size, edge_size,
#                  bias_constant):
#         super(GRU_edge, self).__init__()
        
#         ## Define the architecture
#         self.num_layers = num_layers
#         self.hidden_size = h_edge_size
#         # input
#         self.input = nn.Sequential(
#             nn.Linear(input_size, embedding_size),
#             nn.ReLU()
#         )
#         # gru
#         self.rnn = nn.GRU(input_size=embedding_size, hidden_size=h_edge_size, 
#                           num_layers=num_layers, batch_first=True)
#         # outputs from the gru
#         self.output_to = nn.Sequential(
#                 nn.Linear(h_edge_size, emb_edge_size),
#                 nn.ReLU(),
#                 nn.Linear(emb_edge_size, edge_size)
#             )
#         self.output_from = nn.Sequential(
#                 nn.Linear(h_edge_size, emb_edge_size),
#                 nn.ReLU(),
#                 nn.Linear(emb_edge_size, edge_size)
#             )
#         # initialialization
#         self.hidden = None
#         for name, param in self.rnn.named_parameters():
#             if 'bias' in name:
#                 nn.init.constant_(param, bias_constant)
#             elif 'weight' in name:
#                 nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('sigmoid'))
#                 #nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
#                 #m.weight.data = init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
    
#     def forward(self, input_raw):
#         # input
#         input_egru = self.input(input_raw)
#         # rnn
#         output_raw, self.hidden = self.rnn(input_egru, self.hidden)
#         # output
#         output_from = self.output_from(output_raw)
#         output_to = self.output_to(output_raw)
        
#         return output_from, output_to

    
class GRU_edge_ver2(nn.Module):
    """
    Sequential NN which outputs the edge categories F(i) using GRU_graph hidden state and X(i)
    1 layer at input, rnn with hidden layers, 2 layer at output
    """
    def __init__(self,
                 input_size, embedding_size,
                 h_edge_size, num_layers,
                 emb_edge_size, edge_size,
                 bias_constant):
        super(GRU_edge_ver2, self).__init__()
        
        ## Define the architecture
        self.num_layers = num_layers
        self.hidden_size = h_edge_size
        # input
        self.input = nn.Sequential(
            nn.Linear(input_size, embedding_size),
            nn.ReLU()
        )
        # gru
        self.rnn = nn.GRU(input_size=embedding_size, hidden_size=h_edge_size, 
                          num_layers=num_layers, batch_first=True)
        # outputs from the gru
        self.output= nn.Sequential(
                nn.Linear(h_edge_size, emb_edge_size),
                nn.ReLU(),
                nn.Linear(emb_edge_size, edge_size)
            )
        # initialialization
        self.hidden = None
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, bias_constant)
            elif 'weight' in name:
                #nn.init.xavier_uniform_(param,gain=nn.init.calculate_gain('sigmoid'))
                nn.init.xavier_uniform_(param,gain=nn.init.calculate_gain('relu'))
                #nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                #m.weight.data = init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
                
    
    def forward(self, input_raw):
        # input
        input_egru = self.input(input_raw)
        # rnn
        output_raw, self.hidden = self.rnn(input_egru, self.hidden)
        # output
        output = self.output(output_raw)
        
        return output


class GRU_edge_MHP(nn.Module):
    """
    Sequential NN which outputs M x edge categories F(i) using GRU_graph hidden state and X(i)
    3 layer at input, rnn with hidden layers, 3 layer at output
    """
    def __init__(self,
                 input_size, embedding_size,
                 h_edge_size, num_layers,
                 emb_edge_size, edge_size,
                 bias_constant,
                 number_generators):
        super(GRU_edge_MHP, self).__init__()
        
        ## Define the architecture
        self.num_layers = num_layers
        self.hidden_size = h_edge_size
        self.edge_size = edge_size
        self.num_generators = number_generators
        # input
        self.input = nn.Sequential(
            nn.Linear(input_size, embedding_size),
            nn.ReLU()
        )
        # gru
        self.rnn = nn.GRU(input_size=embedding_size, hidden_size=h_edge_size, 
                          num_layers=num_layers, batch_first=True)
        # outputs from the gru
        self.output= nn.Sequential(
                nn.Linear(h_edge_size, emb_edge_size),
                nn.ReLU(),
                nn.Linear(emb_edge_size, emb_edge_size),
                nn.ReLU(),
                nn.Linear(emb_edge_size, number_generators*edge_size)
            )
        # initialialization
        self.hidden = None
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, bias_constant)
            elif 'weight' in name:
                #nn.init.xavier_uniform_(param,gain=nn.init.calculate_gain('sigmoid'))
                nn.init.xavier_uniform_(param,gain=nn.init.calculate_gain('relu'))
                #nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                #m.weight.data = init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
                
    
    def forward(self, input_raw):
        # input
        input_egru = self.input(input_raw)
        # rnn
        output_raw, self.hidden = self.rnn(input_egru, self.hidden)
        # output
        output = self.output(output_raw)
        output = output.view((output.shape[0], output.shape[1], self.edge_size, self.num_generators))
        
        return output



######## Multiple Hypothesis Prediction Implementation ###############

class MultipleHypothesisPrediction(nn.Module):
    """
    outputs should be (N, E, L, C) and labels should be (N, L)
    N: batch size
    L: sequence length
    E: number of edge categories
    C: number of output generators(corresponding to multiple hypothesis)
    """
    def __init__(self, Loss, num_generators, num_categories, relaxation=0.05, dropout=0.1):
        super(MultipleHypothesisPrediction, self).__init__()
        self.Loss = Loss
        self.num_generators = num_generators
        self.num_categories = num_categories
        self.eps = relaxation
        self.dropout =  dropout

    def forward(self, outputs, labels):
        N, E, L, C = outputs.shape
        N1, L1 = labels.shape
        assert N1==N
        assert L1==L
        labels = labels.view((N,L,1)).expand(-1,-1,C)
        # # dropout
        if self.num_generators>1:
            mask = torch.empty(labels.shape).uniform_(0, 1).to(DEVICE)>self.dropout
            labels = torch.where(mask, labels, -1*torch.ones(labels.shape).long().to(DEVICE))
        # loss for all generators
        loss_vec = self.Loss(outputs, labels)
        # loss of best hypothesis
        M = torch.min(loss_vec, dim=2)[0]
        M_coeff = 1-(self.eps*self.num_generators/(self.num_generators-1))
        # avg loss of all hypotheses
        avg_err = torch.sum(loss_vec, dim=2)
        avg_err_coeff = self.eps/(self.num_generators-1)
        # meta loss
        meta_loss = M_coeff*M + avg_err_coeff*avg_err
        
        return meta_loss


####################### Set Node Generation ##############################################################

class Permute(nn.Module):
    def __init__(self, shape):
        super(Permute, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.permute(*self.shape)


class Object_SetEncoder(nn.Module):
    """
    DeepSets type encoder (set to parameters of latent vector z --> (mu, sigma)).
    Following variants of pooling can be used:
    * sum
    * mean
    * Featurewise Sort (FS)
    """
    def __init__(self, cat_phi_in, cat_phi_hidden, cat_phi_out,
                 count_phi_in, count_phi_hidden, count_phi_out,
                 rho_hidden, rho_out, cardinality_emb_size, 
                 card_embeddding, categ_embedding, count_embedding, 
                 pooling='sum'):
        super(Object_SetEncoder, self).__init__()
        self.card_embeddding = card_embeddding
        self.categ_embedding = categ_embedding
        self.count_embedding = count_embedding
        # pooling
        self.pooling = pooling
        if self.pooling=='FS':
            self.cat_fspool = FSPool(cat_phi_out, 20)
            self.count_fspool = FSPool(count_phi_out, 20)
        ### CARDINALITY ENCODING ######
        self.rho_card = nn.Sequential(
            nn.Linear(cardinality_emb_size, rho_hidden),
            nn.BatchNorm1d(rho_hidden),
            nn.ReLU(),
            nn.Linear(rho_hidden, rho_hidden),
            nn.BatchNorm1d(rho_hidden),
            nn.ReLU(),
            nn.Linear(rho_hidden, rho_hidden),
            nn.BatchNorm1d(rho_hidden),
            nn.ReLU(),
            nn.Linear(rho_hidden, rho_out),
            nn.BatchNorm1d(rho_out),
            nn.ReLU()
        )
        #  ### CATEGORY ENCODING #######
        # elementwise transform for object category
        self.cat_phi = nn.Sequential(
            nn.Conv1d(cat_phi_in, cat_phi_hidden, 1),
            nn.BatchNorm1d(cat_phi_hidden),
            nn.ReLU(),
            nn.Conv1d(cat_phi_hidden, cat_phi_hidden, 1),
            nn.BatchNorm1d(cat_phi_hidden),
            nn.ReLU(),
            nn.Conv1d(cat_phi_hidden, cat_phi_hidden, 1),
            nn.BatchNorm1d(cat_phi_hidden),
            nn.ReLU(),
            nn.Conv1d(cat_phi_hidden, cat_phi_out, 1),
            nn.BatchNorm1d(cat_phi_out),
            nn.ReLU()
        )
        # transform for pooled vector
        self.rho_cat = nn.Sequential(
            nn.Linear(cat_phi_out, rho_hidden),
            nn.BatchNorm1d(rho_hidden),
            nn.ReLU(),
            nn.Linear(rho_hidden, rho_hidden),
            nn.BatchNorm1d(rho_hidden),
            nn.ReLU(),
            nn.Linear(rho_hidden, rho_hidden),
            nn.BatchNorm1d(rho_hidden),
            nn.ReLU(),
            nn.Linear(rho_hidden, rho_out),
            nn.BatchNorm1d(rho_out),
            nn.ReLU()
        )
        #  ### COUNT ENCODING ########
        # elementwise transform for object count
        self.count_phi = nn.Sequential(
            nn.Conv1d(count_phi_in, count_phi_hidden, 1),
            nn.BatchNorm1d(count_phi_hidden),
            nn.ReLU(),
            nn.Conv1d(count_phi_hidden, count_phi_hidden, 1),
            nn.BatchNorm1d(count_phi_hidden),
            nn.ReLU(),
            nn.Conv1d(count_phi_hidden, count_phi_hidden, 1),
            nn.BatchNorm1d(count_phi_hidden),
            nn.ReLU(),
            nn.Conv1d(count_phi_hidden, count_phi_out, 1),
            nn.BatchNorm1d(count_phi_out),
            nn.ReLU()
        )
        # transform for pooled vector
        self.rho_count = nn.Sequential(
            nn.Linear(count_phi_out, rho_hidden),
            nn.BatchNorm1d(rho_hidden),
            nn.ReLU(),
            nn.Linear(rho_hidden, rho_hidden),
            nn.BatchNorm1d(rho_hidden),
            nn.ReLU(),
            nn.Linear(rho_hidden, rho_hidden),
            nn.BatchNorm1d(rho_hidden),
            nn.ReLU(),
            nn.Linear(rho_hidden, rho_out),
            nn.BatchNorm1d(rho_out),
            nn.ReLU()
        )
    def forward(self, encoder_input):
        card, obj_cat, obj_count = encoder_input
        #card_emb = self.card_embeddding(card)
        obj_cat_emb = self.categ_embedding(obj_cat).permute(0,2,1)
        obj_count_emb = self.count_embedding(obj_count).permute(0,2,1)
        # elementwise transformation
        obj_cat_ = self.cat_phi(obj_cat_emb)
        obj_count_ = self.count_phi(obj_count_emb)
        #pooling of category and count separately
        if self.pooling=='sum':
            pooled_obj_cat_ = torch.sum(obj_cat_, dim=2)
            pooled_obj_count_ = torch.sum(obj_count_, dim=2)
        #elif self.pooling=='mean':
        #    pooled_obj_cat_ = torch.mean(obj_cat_, dim=2)
        #    pooled_obj_count_ = torch.mean(obj_count_, dim=2)
        elif self.pooling=='FS':
            pooled_obj_cat_, _ = self.cat_fspool(obj_cat_)
            pooled_obj_count_, _ = self.cat_fspool(obj_count_)
        # feature transformation
        #h_card = self.rho_card(card_emb)
        h_cat = self.rho_cat(pooled_obj_cat_)
        h_count = self.rho_count(pooled_obj_count_)
        # concatenate cardinality, count and category
        #h = torch.cat((h_card, h_cat, h_count), dim=1)
        h = torch.cat((h_cat, h_count), dim=1)
        return h



class Object_SetDecoder(nn.Module):
    """
    Decoder which learns a mapping from latent vector z to cardinality, object categories and their counts.
    """
    def __init__(self, latent_size,
                cat_mlp_hidden, cat_mlp_out, category_dim,
                card_mlp_hidden, card_mlp_out, max_cardinality,
                count_mlp_input, count_mlp_emb,
                count_mlp_hidden, max_count):
        super(Object_SetDecoder, self).__init__()
        self.category_dim = category_dim
        self.max_count = max_count
        self.max_cardinality = max_cardinality
        # object category decoder
        self.cat_mlp = nn.Sequential(
            nn.Linear(latent_size, cat_mlp_hidden),
            nn.BatchNorm1d(cat_mlp_hidden),
            nn.ReLU(),
            nn.Linear(cat_mlp_hidden, cat_mlp_out),
            nn.BatchNorm1d(cat_mlp_out),
            nn.ReLU(),
            nn.Linear(cat_mlp_out, cat_mlp_out),
            nn.BatchNorm1d(cat_mlp_out),
            nn.ReLU(),
            nn.Linear(cat_mlp_out, category_dim),
            nn.Sigmoid()
        )
        # object count decoder
        self.count_hidden = None
        self.count_input = nn.Sequential(
            nn.Linear(count_mlp_input, count_mlp_emb),
            nn.ReLU()
        )
        self.count_rnn = nn.GRU(input_size=count_mlp_emb,
                                hidden_size=latent_size,
                                num_layers=1,
                                batch_first=True)
        self.count_output= nn.Sequential(
            nn.Linear(latent_size, count_mlp_hidden),
            nn.ReLU(),
            nn.Linear(count_mlp_hidden, max_count)
            )
        # self.count_mlp = nn.Sequential(
        #     nn.Linear(count_mlp_input, count_mlp_emb),
        #     Permute((0,2,1)),
        #     nn.BatchNorm1d(count_mlp_emb),
        #     Permute((0,2,1)),
        #     nn.ReLU(),
        #     nn.Linear(count_mlp_emb, count_mlp_hidden),
        #     Permute((0,2,1)),
        #     nn.BatchNorm1d(count_mlp_hidden),
        #     Permute((0,2,1)),
        #     nn.ReLU(),
        #     nn.Linear(count_mlp_hidden, count_mlp_hidden),
        #     Permute((0,2,1)),
        #     nn.BatchNorm1d(count_mlp_hidden),
        #     Permute((0,2,1)),
        #     nn.ReLU(),
        #     nn.Linear(count_mlp_hidden, max_count)
        # )
        # # cardinality decoder
        # self.card_mlp = nn.Sequential(
        #     nn.Linear(latent_size, card_mlp_hidden),
        #     nn.BatchNorm1d(card_mlp_hidden),
        #     nn.ReLU(),
        #     nn.Linear(card_mlp_hidden, card_mlp_out),
        #     nn.BatchNorm1d(card_mlp_out),
        #     nn.ReLU(),
        #     nn.Linear(card_mlp_out, card_mlp_out),
        #     nn.BatchNorm1d(card_mlp_out),
        #     nn.ReLU(),
        #     nn.Linear(card_mlp_out, max_cardinality)
        # )
    # def forward(self, z):
    #     cardinality = self.card_mlp(z)
    #     object_category = self.cat_mlp(z)
    #     object_count = self.count_mlp(z)
    #     object_count = object_count.view((object_count.shape[0], self.max_count, self.category_dim))
    #     return cardinality, object_category, object_count
    # def decode_counts(self, z, decoder_count_mlp_input):
    #     # decode object count
    #     tiled_z = torch.cat([torch.unsqueeze(z, dim=2)]*self.max_cardinality, dim=2)
    #     decoder_count_mlp_input[:, 3*self.category_dim:, :] = tiled_z
    #     decoder_count_mlp_input = decoder_count_mlp_input.permute(0,2,1)
    #     object_count = self.count_mlp(decoder_count_mlp_input).permute(0,2,1)
    #     return object_count
    # def decode_counts_autoregressive(self, z, decoder_count_mlp_input):
    #     # decode object count during generation
    #     decoder_count_mlp_input[0, 3*self.category_dim:, 0] = z
    #     decoder_count_mlp_input = decoder_count_mlp_input.permute(0,2,1)
    #     object_count = self.count_mlp(decoder_count_mlp_input).permute(0,2,1)
    #     return object_count

    def decode_objects(self, z):
        # decode object category
        object_category = self.cat_mlp(z)
        return object_category

    def decode_counts(self, z, count_input):
        # decode object count
        self.count_hidden = z.reshape((1, z.shape[0], z.shape[1]))
        input_ = self.count_input(count_input.permute(0,2,1))
        output_, self.count_hidden = self.count_rnn(input_, self.count_hidden)
        object_count = self.count_output(output_).permute(0,2,1)
        return object_count

    def decode_counts_autoregressive(self, count_input):
        # decode object count
        input_ = self.count_input(count_input.permute(0,2,1))
        output_, self.count_hidden = self.count_rnn(input_, self.count_hidden)
        object_count = self.count_output(output_)
        return object_count

    def forward(self, z, count_input):
        object_category = self.decode_objects(z)
        object_count = self.decode_counts(z, count_input)
        return object_category, object_count



class VAE_ObjectGenerator(nn.Module):
    """
    Variational Autoencoder with a permutation-invariant set encoder (Deep Sets) and a permutation-fixed decoder.
    The data representation is {(y, yc)} = {(man, 3), (table, 1), (chair, 5), (wire, 1), (light, 2)},  Cardinality(m) = 5 (number of elements)
    """
    def __init__(self, set_encoder, set_decoder, encoder_out_size, latent_size, num_latent_samples):
        super(VAE_ObjectGenerator, self).__init__()
        self.num_latent_samples = num_latent_samples
        self.set_encoder = set_encoder
        self.set_decoder = set_decoder
        
        self.fc_mu = nn.Linear(encoder_out_size, latent_size)
        self.fc_var = nn.Sequential(
            nn.Linear(encoder_out_size, latent_size),
            nn.Softplus()
        )
        # random sampler
        self.noise_sampler = torch.distributions.MultivariateNormal(torch.zeros(latent_size), torch.eye(latent_size))


    def reparameterize(self, mu, logvar, couple_eps=False):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        :param mu: (Tensor) Mean of the latent vector
        :param logvar: (Tensor) Standard deviation of the latent vector
        """
        std = torch.exp(0.5 * logvar)
        N, D = mu.shape
        #eps = self.noise_sampler.sample((1,)).to(DEVICE).expand(N, -1)
        if couple_eps:
            eps = self.noise_sampler.sample((self.num_latent_samples,)).to(DEVICE)
            if self.num_latent_samples>1:
                mu = torch.cat([mu]*self.num_latent_samples)
                std = torch.cat([std]*self.num_latent_samples)
                eps = torch.repeat_interleave(eps, N, dim=0)
        else:
            #eps = self.noise_sampler.sample((N,)).to(DEVICE)
            eps = torch.randn_like(std).to(DEVICE)
        #eps = self.noise_sampler.sample((1,)).to(DEVICE)
        return mu + eps*std


    def forward(self, encoder_input, count_input):
        # encode input to hidden vector h
        h = self.set_encoder(encoder_input)
        # get mu and var for latent variable z
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        # sample latent vector z
        z = self.reparameterize(mu, log_var)
        # decode latent vector to output
        cat, count = self.set_decoder(z, count_input)
        output = (cat, count)
        return  output, mu, log_var





############################ Transformer Model #####################################

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class GraphTransformerEncoder(nn.Module):

    def __init__(self, in_size, nhead, hidden_size, nlayers, out_size, dropout=0.1):
        super(GraphTransformerEncoder, self).__init__()
        
        self.pos_encoder = PositionalEncoding(in_size, dropout)
        encoder_layers = TransformerEncoderLayer(in_size, nhead, hidden_size, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.linear = nn.Linear(in_size, out_size)
    def forward(self, src, src_mask=None):
        src = self.pos_encoder(src)
        if src_mask is not None:
            graph_enc = self.transformer_encoder(src, src_mask)
        else:
            graph_enc = self.transformer_encoder(src)
        graph_mem = self.linear(graph_enc)
        return graph_mem


class EdgeTransformerDecoder(nn.Module):

    def __init__(self, in_size, nhead, hidden_size, nlayers, emb_edge_size, edge_size, dropout=0.1):
        super(EdgeTransformerDecoder, self).__init__()
        
        self.pos_encoder = PositionalEncoding(in_size, dropout)
        decoder_layers = TransformerDecoderLayer(in_size, nhead, hidden_size, dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)
        self.linear = nn.Sequential(
                      nn.Linear(in_size, emb_edge_size),
                      nn.ReLU(),
                      nn.Linear(emb_edge_size, edge_size)
                      )

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt = self.pos_encoder(tgt)
        if tgt_mask is not None and memory_mask is not None:
            output = self.transformer_decoder(tgt, memory, tgt_mask, memory_mask)
        else:
            output = self.transformer_decoder(tgt, memory)
        edge_logit = self.linear(output)
        return edge_logit

