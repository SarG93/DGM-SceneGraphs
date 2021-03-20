from model_params import *
from model import *


def get_config(config):
    ordering = config['order']
    weighted_loss = config['class_weight']
    node_pred = config['node_pred']
    edge_pred = config['edge_pred']
    use_argmax = config['use_argmax']
    use_MHP = config['use_MHP']
    return ordering, weighted_loss, node_pred, edge_pred, use_argmax, use_MHP


def set_config(config_params):
    config = {}
    config['order'] = config_params[0]
    config['class_weight'] = config_params[1]
    config['node_pred'] = config_params[2]
    config['edge_pred'] = config_params[3]
    config['use_argmax'] = config_params[4]
    config['use_MHP'] = config_params[5]
    return config


def plot_loss(train_loss, name):
    epochs = np.arange(len(train_loss))
    plt.plot(epochs, train_loss, 'b', label='training loss')
    plt.legend(loc="upper right")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig('./figures/'+ name+ '.png')
    plt.clf()
    plt.cla()
    plt.close()


def instantiate_model_classes(params, config, use_glove=False):
    print('Initialize the model..')
    ordering, weighted_loss, node_pred, edge_pred, use_argmax, use_MHP = get_config(config)
    # Embeddings for input
    node_emb = nn.Embedding(params.num_node_categories,
                            params.node_emb_size, 
                            padding_idx=0,
                            scale_grad_by_freq=False).to(DEVICE) # 0, 1to150
    if use_glove:
        glove_emb = torch.Tensor(pickle.load(open(os.path.join('./data','glove_emb.p'), 'rb')))
        node_emb.load_state_dict({'weight': glove_emb})
    node_emb.weight.requires_grad=False
    edge_emb = nn.Embedding(params.num_edge_categories,
                            params.edge_emb_size,
                            padding_idx=0,
                            scale_grad_by_freq=False).to(DEVICE) # 0, 1to50, 51, 52
    edge_emb.weight.requires_grad=False
    # Node Generator
    if node_pred:
        if use_MHP:
            mlp_node = MLP_node_MHP(h_graph_size=params.mlp_input_size,
                                    embedding_size=params.mlp_emb_size,
                                    node_size=params.mlp_out_size,
                                    num_generators=params.num_generators).to(DEVICE)
        else:
            mlp_node = MLP_node(h_graph_size=params.mlp_input_size,
                                embedding_size=params.mlp_emb_size,
                                node_size=params.mlp_out_size).to(DEVICE)
        gru_graph3 = GRU_graph(max_num_node=params.max_num_node,
                           input_size=params.node_emb_size,
                           embedding_size=params.ggru_emb_size,
                           hidden_size=params.ggru_hidden_size,
                           num_layers=params.ggru_num_layers,
                           bias_constant=params.bias_constant).to(DEVICE)
    else:
        mlp_node = None
        gru_graph3 = None
    # Edge Generator
    if edge_pred:
        gru_graph1 =  GRU_graph(max_num_node=params.max_num_node,
                                input_size=params.ggru_input_size,
                                embedding_size=params.ggru_emb_size,
                                hidden_size=params.ggru_hidden_size,
                                num_layers=params.ggru_num_layers,
                                bias_constant=params.bias_constant).to(DEVICE)
        gru_graph2 =  GRU_graph(max_num_node=params.max_num_node,
                                input_size=params.ggru_input_size,
                                embedding_size=params.ggru_emb_size,
                                hidden_size=params.ggru_hidden_size,
                                num_layers=params.ggru_num_layers,
                                bias_constant=params.bias_constant).to(DEVICE)
        if use_MHP:
            gru_edge1 = GRU_edge_MHP(input_size=params.egru_input_size,
                                    embedding_size=params.egru_emb_input_size,
                                    h_edge_size=params.egru_hidden_size,
                                    num_layers=params.egru_num_layers,
                                    emb_edge_size=params.egru_emb_output_size,
                                    edge_size=params.egru_output_size,
                                    bias_constant=params.bias_constant,
                                    number_generators=params.num_generators).to(DEVICE)
            gru_edge2 = GRU_edge_MHP(input_size=params.egru_input_size,
                                    embedding_size=params.egru_emb_input_size,
                                    h_edge_size=params.egru_hidden_size,
                                    num_layers=params.egru_num_layers,
                                    emb_edge_size=params.egru_emb_output_size,
                                    edge_size=params.egru_output_size,
                                    bias_constant=params.bias_constant,
                                    number_generators=params.num_generators).to(DEVICE)
        else:
            gru_edge1 = GRU_edge_ver2(input_size=params.egru_input_size1,
                                    embedding_size=params.egru_emb_input_size,
                                    h_edge_size=params.egru_hidden_size,
                                    num_layers=params.egru_num_layers,
                                    emb_edge_size=params.egru_emb_output_size,
                                    edge_size=params.egru_output_size,
                                    bias_constant=params.bias_constant).to(DEVICE)
            gru_edge2 = GRU_edge_ver2(input_size=params.egru_input_size2,
                                    embedding_size=params.egru_emb_input_size,
                                    h_edge_size=params.egru_hidden_size,
                                    num_layers=params.egru_num_layers,
                                    emb_edge_size=params.egru_emb_output_size,
                                    edge_size=params.egru_output_size,
                                    bias_constant=params.bias_constant).to(DEVICE)
    else:
        gru_graph1 = None
        gru_graph2 = None
        gru_edge1 = None
        gru_edge2 = None

    return node_emb, edge_emb, mlp_node, gru_graph1, gru_graph2, gru_graph3, gru_edge1, gru_edge2


def save_models(models, path, hyperparam_str, config, params):
    print('Saving model..')
    folder = os.path.join(path, hyperparam_str)
    os.makedirs(folder, exist_ok=True)
    # save config
    config_params = get_config(config)
    ordering, weighted_loss, node_pred, edge_pred, use_argmax, use_MHP = config_params
    pickle.dump(config_params, open(os.path.join(folder,"config.p"), "wb"))
    # save params
    model_params = params.batch_size, params.sample_batches, params.node_lr_init, params.node_lr_end, params.node_lr_decay,\
                   params.edge_lr_init, params.edge_lr_end, params.edge_lr_decay, params.epochs, params.reg, params.bias_constant, params.small_network
    pickle.dump(model_params, open(os.path.join(folder,"params.p"), "wb"))
    # save models
    node_emb, edge_emb, mlp_node, gru_graph1, gru_graph2, gru_graph3, gru_edge1, gru_edge2 = models
    fname = os.path.join(folder, 'node_emb.dat')
    torch.save(node_emb.state_dict(), fname)
    fname = os.path.join(folder, 'edge_emb.dat')
    torch.save(edge_emb.state_dict(), fname)
    if node_pred:
        fname = os.path.join(folder, 'MLP_node.dat')
        torch.save(mlp_node.state_dict(), fname)
        fname = os.path.join(folder, 'GRU_graph3.dat')
        torch.save(gru_graph3.state_dict(), fname)
    if edge_pred:
        fname = os.path.join(folder, 'GRU_graph1.dat')
        torch.save(gru_graph1.state_dict(), fname)
        fname = os.path.join(folder, 'GRU_graph2.dat')
        torch.save(gru_graph2.state_dict(), fname)
        fname = os.path.join(folder, 'GRU_edge1.dat')
        torch.save(gru_edge1.state_dict(), fname)
        fname = os.path.join(folder, 'GRU_edge2.dat')
        torch.save(gru_edge2.state_dict(), fname)
    

def load_models(path, hyperparam_str):
    print('Loading trained model..')
    folder = os.path.join(path, hyperparam_str)
    # load config
    config_params = pickle.load(open(os.path.join(folder,"config.p"), 'rb'))
    ordering, weighted_loss, node_pred, edge_pred, use_argmax, use_MHP = config_params
    config = set_config(config_params)
    # load params
    model_params = pickle.load(open(os.path.join(folder,"params.p"), 'rb'))
    batch_size, sample_batches, node_lr_init, node_lr_end, node_lr_decay, edge_lr_init, edge_lr_end, edge_lr_decay, num_epochs, reg, bias_constant, small_network = model_params
    params = Model_params(batch_size, sample_batches, node_lr_init, node_lr_end, node_lr_decay,
                            edge_lr_init, edge_lr_end, edge_lr_decay, num_epochs, reg, bias_constant,
                            config, small_network)
    # load models
    if USE_TRANSFORMER:
        models = instantiate_model_classes_transformer(params, config)
    else:
        models = instantiate_model_classes(params, config)
    node_emb, edge_emb, mlp_node, gru_graph1, gru_graph2, gru_graph3, gru_edge1, gru_edge2 = models
    fname = os.path.join(folder, 'node_emb.dat')
    node_emb.load_state_dict(torch.load(fname, map_location=DEVICE))
    fname = os.path.join(folder, 'edge_emb.dat')
    edge_emb.load_state_dict(torch.load(fname, map_location=DEVICE))
    if node_pred:
        fname = os.path.join(folder, 'MLP_node.dat')
        mlp_node.load_state_dict(torch.load(fname, map_location=DEVICE))
        fname = os.path.join(folder, 'GRU_graph3.dat')
        gru_graph3.load_state_dict(torch.load(fname, map_location=DEVICE))
    if edge_pred:
        fname = os.path.join(folder, 'GRU_graph1.dat')
        gru_graph1.load_state_dict(torch.load(fname, map_location=DEVICE))
        fname = os.path.join(folder, 'GRU_graph2.dat')
        gru_graph2.load_state_dict(torch.load(fname, map_location=DEVICE))
        fname = os.path.join(folder, 'GRU_edge1.dat')
        gru_edge1.load_state_dict(torch.load(fname, map_location=DEVICE))
        fname = os.path.join(folder, 'GRU_edge2.dat')
        gru_edge2.load_state_dict(torch.load(fname, map_location=DEVICE))

    models = node_emb, edge_emb, mlp_node, gru_graph1, gru_graph2, gru_graph3, gru_edge1, gru_edge2
    return params, config, models


########################## NODE SET GENERATION #########################################################

def instantiate_setVAE_models(params, use_transformer=False):
    """
    Instantiate models for set node generation using VAE
    """
    cardinality_emb = nn.Embedding(params.max_cardinality,
                                params.cardinality_emb_size, 
                                scale_grad_by_freq=False).to(DEVICE)
    category_emb = nn.Embedding(params.category_dim+1,
                                params.category_emb_size, 
                                padding_idx=0,
                                scale_grad_by_freq=False).to(DEVICE)
    count_emb = nn.Embedding(params.max_count+1,
                            params.count_emb_size, 
                            padding_idx=0,
                            scale_grad_by_freq=False).to(DEVICE)
    set_encoder = Object_SetEncoder(cat_phi_in=params.cat_phi_in,
                                    cat_phi_hidden=params.cat_phi_hidden,
                                    cat_phi_out=params.cat_phi_out,
                                    count_phi_in=params.count_phi_in,
                                    count_phi_hidden=params.count_phi_hidden,
                                    count_phi_out=params.count_phi_out,
                                    rho_hidden=params.rho_hidden,
                                    rho_out=params.rho_out,
                                    cardinality_emb_size= params.cardinality_emb_size,
                                    card_embeddding = cardinality_emb,
                                    categ_embedding = category_emb,
                                    count_embedding = count_emb).to(DEVICE)
    set_decoder = Object_SetDecoder(latent_size=params.latent_size, 
                                    cat_mlp_hidden=params.cat_mlp_hidden,
                                    cat_mlp_out=params.cat_mlp_out,
                                    category_dim=params.category_dim,
                                    card_mlp_hidden=params.card_mlp_hidden,
                                    card_mlp_out=params.card_mlp_out,
                                    max_cardinality=params.max_cardinality,
                                    count_mlp_input=params.count_mlp_input,
                                    count_mlp_emb=params.count_mlp_emb,
                                    count_mlp_hidden=params.count_mlp_hidden,
                                    max_count=params.max_count
                                    ).to(DEVICE)
    VAE_model = VAE_ObjectGenerator(set_encoder,
                                    set_decoder,
                                    encoder_out_size=2*params.rho_out,
                                    latent_size=params.latent_size,
                                    num_latent_samples=params.num_latent_samples).to(DEVICE)

    return (cardinality_emb, category_emb, count_emb, set_encoder, set_decoder, VAE_model)


def save_VAE_model(models, path, hyperparam_str, params):
    print('Saving model..')
    folder = os.path.join(path, hyperparam_str)
    os.makedirs(folder, exist_ok=True)
    # save params
    model_params = params.batch_size, params.lr, params.epochs
    pickle.dump(model_params, open(os.path.join(folder,"params.p"), "wb"))
    # save models
    cardinality_emb, category_emb, count_emb, set_encoder, set_decoder, VAE_model = models
    fname = os.path.join(folder, 'cardinality_emb.dat')
    torch.save(cardinality_emb.state_dict(), fname)
    fname = os.path.join(folder, 'category_emb.dat')
    torch.save(category_emb.state_dict(), fname)
    fname = os.path.join(folder, 'count_emb.dat')
    torch.save(count_emb.state_dict(), fname)
    fname = os.path.join(folder, 'set_encoder.dat')
    torch.save(set_encoder.state_dict(), fname)
    fname = os.path.join(folder, 'set_decoder.dat')
    torch.save(set_decoder.state_dict(), fname)
    fname = os.path.join(folder, 'VAE_model.dat')
    torch.save(VAE_model.state_dict(), fname)


def load_VAE_model(path, hyperparam_str):
    print('Loading trained model..')
    folder = os.path.join(path, hyperparam_str)
    # load params
    model_params = pickle.load(open(os.path.join(folder,"params.p"), 'rb'))
    batch_size, lr, num_epochs = model_params
    params = Model_params_VAE(batch_size, lr, num_epochs)
    # load models
    models = instantiate_setVAE_models(params)
    cardinality_emb, category_emb, count_emb, set_encoder, set_decoder, VAE_model = models
    fname = os.path.join(folder, 'cardinality_emb.dat')
    cardinality_emb.load_state_dict(torch.load(fname, map_location=DEVICE))
    fname = os.path.join(folder, 'category_emb.dat')
    category_emb.load_state_dict(torch.load(fname, map_location=DEVICE))
    fname = os.path.join(folder, 'count_emb.dat')
    count_emb.load_state_dict(torch.load(fname, map_location=DEVICE))
    fname = os.path.join(folder, 'set_encoder.dat')
    set_encoder.load_state_dict(torch.load(fname, map_location=DEVICE))
    fname = os.path.join(folder, 'set_decoder.dat')
    set_decoder.load_state_dict(torch.load(fname, map_location=DEVICE))
    fname = os.path.join(folder, 'VAE_model.dat')
    VAE_model.load_state_dict(torch.load(fname, map_location=DEVICE))
    models = cardinality_emb, category_emb, count_emb, set_encoder, set_decoder, VAE_model
    return params, models



def instantiate_model_classes_transformer(params, config, use_glove=False):
    print('Initialize the model..')
    ordering, weighted_loss, node_pred, edge_pred, use_argmax, use_MHP = get_config(config)
    
    # Embeddings for input
    node_emb = nn.Embedding(params.num_node_categories,
                            params.node_emb_size, 
                            padding_idx=0,
                            scale_grad_by_freq=False).to(DEVICE) # 0, 1to150
    if use_glove:
        glove_emb = torch.Tensor(pickle.load(open(os.path.join('./data','glove_emb.p'), 'rb')))
        node_emb.load_state_dict({'weight': glove_emb})
    node_emb.weight.requires_grad=False
    #print(node_emb)
    edge_emb = nn.Embedding(params.num_edge_categories,
                            params.edge_emb_size,
                            padding_idx=0,
                            scale_grad_by_freq=False).to(DEVICE)# 0, 1to50, 51, 52
    edge_emb.weight.requires_grad=False
    #print(edge_emb)
    # Graph Encoder
    GTE3 = GraphTransformerEncoder(in_size=params.ggru_input_size,
                                         nhead=4,
                                         hidden_size=1024,
                                         nlayers=3,
                                         out_size=256).to(DEVICE)
    GTE1 =  GraphTransformerEncoder(in_size=params.ggru_input_size,
                                         nhead=4,
                                         hidden_size=1024,
                                         nlayers=3, 
                                         out_size=params.egru_input_size).to(DEVICE)
    GTE2 =  GraphTransformerEncoder(in_size=params.ggru_input_size,
                                         nhead=4,
                                         hidden_size=1024,
                                         nlayers=3,
                                         out_size=params.egru_input_size).to(DEVICE)
    #print(GTE1)
    #print(GTE2)
    #print(GTE3)
    # Node Decoder
    mlp_node = MLP_node(h_graph_size=256,
                        embedding_size=params.mlp_emb_size,
                        node_size=params.mlp_out_size).to(DEVICE)
    #print(mlp_node)
    # Edge Decoder
    ETD1 = EdgeTransformerDecoder(in_size=params.egru_input_size,
                                nhead=4,
                                hidden_size=512,
                                nlayers=2, 
                                emb_edge_size=params.egru_emb_output_size, 
                                edge_size=params.egru_output_size).to(DEVICE)
    
    ETD2 = EdgeTransformerDecoder(in_size=params.egru_input_size,
                                nhead=4,
                                hidden_size=512, 
                                nlayers=2, 
                                emb_edge_size=params.egru_emb_output_size, 
                                edge_size=params.egru_output_size).to(DEVICE)

    return node_emb, edge_emb, mlp_node, GTE1, GTE2, GTE3, ETD1, ETD2
