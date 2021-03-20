from train_helper import *
from data import *

def change_format(graph, ind_to_classes, ind_to_predicates,):
    X, F = graph
    X = list(X)
    F[F==51] = 0
    objs = [ind_to_classes[x-1] for x in X]
    to_idx_lst, from_idx_lst = F.nonzero()
    triples = list()
    for to_idx, from_idx in zip(to_idx_lst, from_idx_lst):
        triples.append([int(to_idx), ind_to_predicates[int(F[to_idx, from_idx])-1], int(from_idx)])
    return objs, triples


def draw_scene_graph(graph, ind_to_classes, ind_to_predicates):
    """
    Use GraphViz to draw a scene graph. If vocab is not passed then we assume
    that objs and triples are python lists containing strings for object and
    relationship names.
    Using this requires that GraphViz is installed. On Ubuntu 16.04 this is easy:
    sudo apt-get install graphviz
    """
    objs, triples = change_format(graph, ind_to_classes, ind_to_predicates,)
    #output_filename = kwargs.pop('output_filename', 'graph.png')
    orientation = 'V' #kwargs.pop('orientation', 'V')
    edge_width = 6# kwargs.pop('edge_width', 6)
    arrow_size = 1.5 #kwargs.pop('arrow_size', 1.5)
    binary_edge_weight = 1.2 #kwargs.pop('binary_edge_weight', 1.2)
    
    if orientation not in ['V', 'H']:
        raise ValueError('Invalid orientation "%s"' % orientation)
    rankdir = {'H': 'LR', 'V': 'TD'}[orientation]

    # General setup, and style for object nodes
    lines = [
        'digraph{',
        'graph [size="5,3",ratio="compress",dpi="300",bgcolor="transparent"]',
        'rankdir=%s' % rankdir,
        'nodesep="0.5"',
        'ranksep="0.5"',
        'node [shape="box",style="rounded,filled",fontsize="48",color="none"]',
        'node [fillcolor="lightpink1"]',
    ]
    # Output nodes for objects
    for i, obj in enumerate(objs):
        lines.append('%d [label="%s"]' % (i, obj))

    # Output relationships
    next_node_id = len(objs)
    lines.append('node [fillcolor="lightblue1"]')
    for s, p, o in triples:
        lines += [
        '%d [label="%s"]' % (next_node_id, p),
        '%d->%d [penwidth=%f,arrowsize=%f,weight=%f]' % (
            s, next_node_id, edge_width, arrow_size, binary_edge_weight),
        '%d->%d [penwidth=%f,arrowsize=%f,weight=%f]' % (
            next_node_id, o, edge_width, arrow_size, binary_edge_weight)
        ]
        next_node_id += 1
    lines.append('}')
    
    ff, dot_filename = tempfile.mkstemp()
    with open(dot_filename, 'w') as f:
        for line in lines:
            f.write('%s\n' % line)
    os.close(ff)

    # Shell out to invoke graphviz; this will save the resulting image to disk,
    # so we read it, delete it, then return it.
    output_filename = '/home/snowcrash/CSEThesis/thesis/dgm_scenegraphs/data/temp_sg.png'
    output_format = os.path.splitext(output_filename)[1][1:]
    os.system('dot -T%s %s > %s' % (output_format, dot_filename, output_filename))
    os.remove(dot_filename)
    img = imread(output_filename)
    os.remove(output_filename)
    
    return img




def vis_graphs(params, X, Fto, Ffrom, ind_to_classes, ind_to_predicates):
    sg = Digraph('sg', format='png')
    for idx, node in enumerate(X-1):
        sg.node(str(idx), ind_to_classes[node])
    
    Fto = np.where(Fto==params.no_edge_token, 0, Fto)
    Fto = np.where(Fto==51, 0, Fto)
    to_edge_obj, to_edge_subj = np.nonzero(Fto)
    for obj, subj in zip(to_edge_obj, to_edge_subj):
        #print(obj, subj, Fto[obj, subj])
        sg.edge(str(obj), str(subj), label=ind_to_predicates[int(Fto[obj, subj]-1)])
    
    Ffrom = np.where(Ffrom==params.no_edge_token, 0, Ffrom)
    Ffrom = np.where(Ffrom==51, 0, Ffrom)
    from_edge_subj, from_edge_obj = np.nonzero(Ffrom)
    for obj, subj in zip(from_edge_obj, from_edge_subj):
        sg.edge(str(obj), str(subj), label=ind_to_predicates[int(Ffrom[subj, obj]-1)])

    # sg = Digraph('sg', format='png')
    # for idx, node in enumerate(X):
    #     sg.node(str(idx), ind_to_classes[node])
    
    # Fto = np.where(Fto==params.no_edge_token, 0, Fto)
    # to_edge_obj, to_edge_subj = np.nonzero(Fto)
    # for obj, subj in zip(to_edge_obj, to_edge_subj):
    #     sg.edge(str(obj), str(subj), label=ind_to_predicates[int(Fto[obj, subj])])
    
    # Ffrom = np.where(Ffrom==params.no_edge_token, 0, Ffrom)
    # from_edge_subj, from_edge_obj = np.nonzero(Ffrom)
    # for obj, subj in zip(from_edge_obj, from_edge_subj):
    #     sg.edge(str(obj), str(subj), label=ind_to_predicates[int(Ffrom[subj, obj])])
    
    return sg


# ------------------- Full graph Generation ------------------------------------------------
def prior_distribution(graphs, ordering):
    class_dict = dict()
    for _ in range(10):
        for graph in graphs:
            X, F = graph
            # order the graph to a sequence to get permuted X and F
            if ordering=='random':
                X, F = random_ordered(X, F)
            elif ordering == 'predefined':
                X, F = predefined_ordered(X, F)
            elif ordering == 'none':
                pass
            elif ordering=='bfs':
                root =  random.choice(np.arange(X.shape[0]))     
                Xin, Fin = bfs_ordered(X, F, root)
            elif ordering=='3dbfs':
                root = list(X).index(32)
                X, F = bfs_ordered(X, F, root)
            elif ordering=='motif_based':
                X, F = motif_based_ordered(X, F, IND_TO_CLASSES)
            else:
                raise print('Ordering not understood')
            
            first_node = X[0]
            if first_node not in class_dict:
                class_dict[first_node] = 1
            else:
                class_dict[first_node] += 1

    total = sum(list(class_dict.values()))
    class_dict = {k:v/total for k, v in class_dict.items()}
    print(sum(list(class_dict.values())))

    return dict(sorted(class_dict.items())), total



def generate_one_graph(params, config, models, first_node, edge_supervision=True):
    num_graphs=1
    ordering, weighted_loss, node_pred, edge_pred, use_argmax, use_MHP = get_config(config)
    node_emb, edge_emb, mlp_node, gru_graph1, gru_graph2, gru_graph3, gru_edge1, gru_edge2 = models
    # set models to eval mode
    node_emb.eval()
    edge_emb.eval()
    mlp_node.eval()
    gru_graph3.eval()
    gru_graph1.eval()
    gru_graph2.eval()
    gru_edge1.eval()
    gru_edge2.eval()
    # intantiate generated graphs
    X = torch.zeros(num_graphs,
                    params.max_num_node-1).to(DEVICE).long()
    Fto = torch.zeros(num_graphs,
                      params.max_num_node-1,
                      params.max_num_node).to(DEVICE).long()
    Ffrom = torch.zeros(num_graphs,
                      params.max_num_node-1,
                      params.max_num_node).to(DEVICE).long()
    # sample initial object
    Xsample = torch.Tensor([first_node-1]).long().to(DEVICE)
    # initial edge vector
    init_edges = torch.zeros(num_graphs, params.max_num_node-1).to(DEVICE).long()
    edge_SOS_token = torch.Tensor([params.edge_SOS_token]).to(DEVICE).long()
    # init gru_graph hidden state
    gru_graph1.hidden = gru_graph1.init_hidden(num_graphs)
    gru_graph2.hidden = gru_graph2.init_hidden(num_graphs)
    gru_graph3.hidden = gru_graph3.init_hidden(num_graphs)
    
    softmax_MHP = nn.Softmax(dim=1)
    softmax = nn.Softmax(dim=0)
    for i in range(params.max_num_node-1):
        # update graph info with generated nodes/edges
        X[:,i] = Xsample+1
        Fto_vec = Fto[:,:,i]
        Ffrom_vec = Ffrom[:,:,i]

        Xsample1  = torch.unsqueeze(X[:,i] ,1)
        Fto_vec  = torch.unsqueeze(Fto_vec, 1)
        Ffrom_vec  = torch.unsqueeze(Ffrom_vec, 1)
        Xsample1 = node_emb(Xsample1)
        Fto_vec = edge_emb(Fto_vec)
        Fto_vec = Fto_vec.contiguous().view(Fto_vec.shape[0],
                                            Fto_vec.shape[1], -1)
        Ffrom_vec = edge_emb(Ffrom_vec)
        Ffrom_vec = Ffrom_vec.contiguous().view(Ffrom_vec.shape[0],
                                                Ffrom_vec.shape[1], -1)
        gru_graph_in = torch.cat((Xsample1.float(), Fto_vec.float(), Ffrom_vec.float()), 2)

        # run one step of gru_graph
        gru_edge_hidden1 = gru_graph1(gru_graph_in, list(np.ones(num_graphs))).data
        gru_edge_hidden2 = gru_graph2(gru_graph_in, list(np.ones(num_graphs))).data
        mlp_input = gru_graph3(Xsample1.float(), list(np.ones(num_graphs))).data

        # run mlp_node and sample next object
        Xscores = mlp_node(mlp_input)
        if use_argmax:
            Xsample = torch.argmax(Xscores, dim=2)
        else:
            if use_MHP:
                Xscores = torch.squeeze(Xscores).permute(1,0)
                node_rand_idx = int(np.random.choice(params.num_generators, 1))
                Xsample = torch.multinomial(softmax_MHP(Xscores), 1)[node_rand_idx]
            else:
                Xscores = torch.squeeze(Xscores)
                Xsample = torch.multinomial(softmax(Xscores), 1)
        # exit if EOS token is encountered
        if Xsample.data.cpu().numpy() == params.node_EOS_token or i==params.max_num_node-2:
            if i==0 or i==1:
                scene_graph = None
                break
            else:
                X = X[:,0:i]
                Fto = Fto[:,0:i,0:i]
                Ffrom = Ffrom[:,0:i,0:i]
                X_gen = torch.squeeze(X).cpu().numpy()
                Fto_gen = torch.squeeze(Fto).cpu().numpy()
                Ffrom_gen = torch.squeeze(Ffrom).cpu().numpy()
                scene_graph = X_gen, Fto_gen, Ffrom_gen
                break

        # get initial hidden state of gru_edge
        if params.egru_num_layers>1:
            gru_edge_hidden1 = torch.cat((gru_edge_hidden1,
                                          torch.zeros(params.egru_num_layers-1,
                                                      gru_edge_hidden1.shape[1],
                                                      gru_edge_hidden1.shape[2]).to(DEVICE)), 0)
            gru_edge_hidden2 = torch.cat((gru_edge_hidden2,
                                        torch.zeros(params.egru_num_layers-1,
                                                    gru_edge_hidden2.shape[1],
                                                    gru_edge_hidden2.shape[2]).to(DEVICE)), 0)
        gru_edge1.hidden = gru_edge_hidden1
        gru_edge2.hidden = gru_edge_hidden2

        # init edge vectors
        Fto_vec = init_edges.clone()
        Ffrom_vec = init_edges.clone()
        for j in range(i+1):
            # input for gru_in
            x1=X[:,j]
            x2=Xsample+1
            fto = Fto_vec[:,j-1] if j>0 else edge_SOS_token
            ffrom = Ffrom_vec[:,j-1] if j>0 else edge_SOS_token
            # print('Inputs to egru', x1, x2, fto, ffrom)
            x1=node_emb(x1.view(x1.shape[0], 1))
            x2=node_emb(x2.view(x2.shape[0], 1))
            fto = edge_emb(fto.view(fto.shape[0], 1))
            ffrom = edge_emb(ffrom.view(ffrom.shape[0], 1))
            # run gru_edge and sample next edge
            if edge_supervision:
                gru_edge_in1 = torch.cat((x1, x2, fto, ffrom), 2)
            else:
                gru_edge_in1 = torch.cat((fto, ffrom), 2)
            Fto_scores = gru_edge2(gru_edge_in1)
            Fto_scores = torch.squeeze(Fto_scores)
            Fto_sample = torch.multinomial(softmax(Fto_scores), 1)
            Fto_vec[:, j] = torch.squeeze(Fto_sample)+1
            fto_out = Fto_vec[:, j]
            fto_out = edge_emb(fto_out.view(fto_out.shape[0], 1))
            if edge_supervision:
                if EDGE_INFO:
                    gru_edge_in2 = torch.cat((x1, x2, fto, ffrom, fto_out), 2)
                else:
                    gru_edge_in2 = torch.cat((x1, x2, fto, ffrom), 2)
            else:
                gru_edge_in2 = torch.cat((fto, ffrom), 2)
            Ffrom_scores = gru_edge1(gru_edge_in2)
            Ffrom_scores = torch.squeeze(Ffrom_scores)
            Ffrom_sample = torch.multinomial(softmax(Ffrom_scores), 1)
            Ffrom_vec[:, j] = torch.squeeze(Ffrom_sample)+1
            # print('generated edges ', Fto_sample, Ffrom_sample)
            # update hidden state of gru_edge
            gru_edge1.hidden = gru_edge1.hidden.data.to(DEVICE)
            gru_edge2.hidden = gru_edge2.hidden.data.to(DEVICE)
        
        # update hidden state of gru_graph
        gru_graph1.hidden = gru_graph1.hidden.data.to(DEVICE)
        gru_graph2.hidden = gru_graph2.hidden.data.to(DEVICE)
        gru_graph3.hidden = gru_graph3.hidden.data.to(DEVICE)
        Fto[:,:,i+1] = Fto_vec
        Ffrom[:,:,i+1] = Ffrom_vec
    #print(scene_graph)
    return scene_graph



def generate_one_graph_transformer(params, config, models, first_node, ind_to_classes, ind_to_predicates, edge_supervision=True):
    num_graphs=1
    ordering, weighted_loss, node_pred, edge_pred, use_argmax, use_MHP = get_config(config)
    node_emb, edge_emb, mlp_node, GTE1, GTE2, GTE3, ETD1, ETD2 = models
    # set models to eval mode
    node_emb.eval()
    edge_emb.eval()
    mlp_node.eval()
    GTE3.eval()
    GTE1.eval()
    GTE2.eval()
    ETD1.eval()
    ETD2.eval()
    # intantiate generated graphs
    X = torch.zeros(num_graphs, params.max_num_node-1).to(DEVICE).long()
    Fto = torch.zeros(num_graphs, params.max_num_node-1, params.max_num_node-1).to(DEVICE).long()
    Ffrom = torch.zeros(num_graphs, params.max_num_node-1, params.max_num_node-1).to(DEVICE).long()
    # sample initial object
    Xsample = torch.Tensor([first_node-1]).long().to(DEVICE)
    X[:, 0] = Xsample+1
    print('Node: ', ind_to_classes[torch.squeeze(Xsample).cpu().numpy()])
    #print(Xsample, ind_to_classes[Xsample])
    # initial edge vector
    init_edges = torch.zeros(num_graphs, params.max_num_node-1).to(DEVICE).long()
    edge_SOS_token = torch.Tensor([params.edge_SOS_token]).to(DEVICE).long()
    
    softmax_MHP = nn.Softmax(dim=1)
    softmax = nn.Softmax(dim=0)
    src_mask = generate_square_subsequent_mask(params.max_num_node-1).to(DEVICE)
    scene_graph = None
    for i in range(params.max_num_node-2):

        # update graph info with generated nodes/edges
        #X[:,i] = Xsample+1
        #Fto_vec = Fto[:,:,i]
        #Ffrom_vec = Ffrom[:,:,i]
        # Input for GTE
        #Xsample1  = node_emb(torch.unsqueeze(X[:,i] ,1))
        #Fto_vec  = edge_emb(torch.unsqueeze(Fto_vec, 1))
        #Ffrom_vec  = edge_emb(torch.unsqueeze(Ffrom_vec, 1))
        #Fto_vec = Fto_vec.contiguous().view(Fto_vec.shape[0], Fto_vec.shape[1], -1)
        #Ffrom_vec = Ffrom_vec.contiguous().view(Ffrom_vec.shape[0], Ffrom_vec.shape[1], -1)
        Xin = node_emb(X)
        Fto_in = edge_emb(Fto)
        Fto_in = Fto_in.contiguous().view(Fto_in.shape[0], Fto_in.shape[1], -1)
        Ffrom_in = edge_emb(Ffrom)
        Ffrom_in = Ffrom_in.contiguous().view(Ffrom_in.shape[0], Ffrom_in.shape[1], -1)
        GTE_in = torch.cat((Xin, Fto_in, Ffrom_in), 2).permute(1, 0, 2)
        #print(GTE_in.squeeze()[:,0])
        # run one step of GTE
        GTE_mem1 = GTE1(GTE_in, src_mask=src_mask)[i]
        GTE_mem2 = GTE2(GTE_in, src_mask=src_mask)[i]
        GTE_mem3 = GTE3(GTE_in, src_mask=src_mask)[i]
        # run mlp_node and sample next object
        Xscores = mlp_node(GTE_mem3)
        Xscores = torch.squeeze(Xscores)
        Xsample = torch.multinomial(softmax(Xscores), 1)
        X[:,i+1] = Xsample+1
        #print('Node: ')
        #print(softmax(Xscores))
        #print('Edges')
        # exit if EOS token is encountered
        if Xsample.data.cpu().numpy() == params.node_EOS_token:
            if i>1:
                X = X[:,0:i+1]
                Fto = Fto[:,0:i+1,0:i+1]
                Ffrom = Ffrom[:,0:i+1,0:i+1]
                X_gen = torch.squeeze(X).cpu().numpy()
                Fto_gen = torch.squeeze(Fto).cpu().numpy()
                Ffrom_gen = torch.squeeze(Ffrom).cpu().numpy()
                scene_graph = X_gen, Fto_gen, Ffrom_gen
            break
        print('Node: ', torch.squeeze(Xsample).cpu().numpy(), ind_to_classes[torch.squeeze(Xsample).cpu().numpy()])
        #print(Xsample, ind_to_classes[Xsample])
        # init edge vectors
        Fto_vec = init_edges.clone()
        Fto_vec[:, 0] = edge_SOS_token
        Ffrom_vec = init_edges.clone()
        Ffrom_vec[:, 0] = edge_SOS_token
        X1_vec = init_edges.clone()
        X2_vec = init_edges.clone()
        for j in range(i+1):
            # input for ETD
            X1_vec[:, j] = X[:,j]
            X2_vec[:, j] = X[:,i+1]
            x1=node_emb(X1_vec)
            x2=node_emb(X2_vec)
            fto = edge_emb(Fto_vec)
            ffrom = edge_emb(Ffrom_vec)
            gru_edge_in = torch.cat((x1, x2, fto, ffrom), 2).permute(1, 0, 2)
            #print(gru_edge_in.squeeze()[:,0])
            # run gru_edge and sample next edge
            Ffrom_scores = ETD1(gru_edge_in, GTE_mem1, tgt_mask=src_mask).squeeze()[i]
            Ffrom_sample = torch.multinomial(softmax(Ffrom_scores), 1)
            #print(softmax(Ffrom_scores))
            #print(Ffrom_sample)
            Fto_scores = ETD2(gru_edge_in, GTE_mem2, tgt_mask=src_mask).squeeze()[i]
            Fto_sample = torch.multinomial(softmax(Fto_scores), 1)

            Fto_ = torch.squeeze(Fto_sample).cpu().numpy()
            Ffrom_ = torch.squeeze(Ffrom_sample).cpu().numpy()
            if Ffrom_!=50:
                print('Edge: ', ind_to_classes[torch.squeeze(X[:,j]).cpu().numpy()-1], 
                                ind_to_classes[torch.squeeze(X[:,i+1]).cpu().numpy()-1],
                                torch.squeeze(Ffrom_sample).cpu().numpy(), ind_to_predicates[torch.squeeze(Ffrom_sample).cpu().numpy()])
            if Fto_!=50:
                print('Edge: ', ind_to_classes[torch.squeeze(X[:,j]).cpu().numpy()-1], 
                                ind_to_classes[torch.squeeze(X[:,i+1]).cpu().numpy()-1],
                                torch.squeeze(Fto_sample).cpu().numpy(), ind_to_predicates[torch.squeeze(Fto_sample).cpu().numpy()])            
            #print(softmax(Fto_scores))
            #print(Fto_sample)
            #print()
            #print(Ffrom_sample, Fto_sample)
            # update edge_vec history
            Fto_vec[:, j] = torch.squeeze(Fto_sample)+1
            Ffrom_vec[:, j] = torch.squeeze(Ffrom_sample)+1
        print()
        # update edge_matrix history
        Fto[:,:,i+1] = Fto_vec
        Ffrom[:,:,i+1] = Ffrom_vec

        #X_gen = torch.squeeze(X[:,0:i+2], 0).cpu().numpy()
        #print(X_gen)
        #Fto_gen = torch.squeeze(Fto[:,0:i+2,0:i+2], 0).cpu().numpy()
        #print(Fto_gen)
        #Ffrom_gen = torch.squeeze(Ffrom[:,0:i+2,0:i+2], 0).cpu().numpy()
        #print(Ffrom_gen)
        #print()
        #sg = vis_graphs(params, X_gen, Fto_gen, Ffrom_gen, ind_to_classes, ind_to_predicates)
        #sg.render(os.path.join('/home/snowcrash/CSEThesis/thesis/dgm_scenegraphs/selected_figures/transformer/1', str(i)), view=False)
    return scene_graph



def generate_scene_graphs(path, params, config, models, num_graphs, class_dict, ind_to_classes, ind_to_predicates, make_visuals=True):
    os.makedirs(path, exist_ok=True)
    graphs = []
    first_node_list = np.random.choice(list(class_dict.keys()), num_graphs, replace=True, p=list(class_dict.values()))
    for idx, first_node in zip(range(num_graphs), first_node_list):
        print(idx, first_node)
        if USE_TRANSFORMER:
            graph = generate_one_graph_transformer(params, config, models, first_node, ind_to_classes, ind_to_predicates)
        else:
            graph = generate_one_graph(params, config, models, first_node)
        if graph is not None:
            X, Fto, Ffrom = graph
            F = Fto + np.transpose(Ffrom)
            graphs.append([X, F])
            if make_visuals:
                #sg = vis_graphs(params, X, Fto, Ffrom, ind_to_classes, ind_to_predicates)
                #sg.render(os.path.join(path, str(idx)), view=False)
                sg = draw_scene_graph([X, F], ind_to_classes, ind_to_predicates)
                imwrite(os.path.join(path, str(idx)+'.png'), sg)
    # save dataset
    pickle.dump(graphs, open(os.path.join(path, 'data_gen.p'), 'wb'))
    return graphs


# ------------- Conditional edge generation -------------------------------------
def generate_edges_given_nodes(params, config, Xin, models, edge_supervision=True):
    scene_graph = None
    num_graphs=1
    ordering, weighted_loss, node_pred, edge_pred, use_argmax, use_MHP = get_config(config)
    node_emb, edge_emb, mlp_node, gru_graph1, gru_graph2, gru_graph3, gru_edge1, gru_edge2 = models
    # set models to eval model
    node_emb.eval()
    edge_emb.eval()
    gru_graph1.eval()
    gru_graph2.eval()
    gru_edge1.eval()
    gru_edge2.eval()
        
    # intantiate generated graphs
    X = torch.zeros(num_graphs,
                    params.max_num_node-1).to(DEVICE).long()
    X[:, 0:Xin.shape[0]] = torch.Tensor(Xin).to(DEVICE).long()
    Fto = torch.zeros(num_graphs,
                      params.max_num_node-1,
                      params.max_num_node).to(DEVICE).long()
    Ffrom = torch.zeros(num_graphs,
                      params.max_num_node-1,
                      params.max_num_node).to(DEVICE).long()
    # initial edge vector
    init_edges = torch.zeros(num_graphs, params.max_num_node-1).to(DEVICE).long()
    edge_SOS_token = torch.Tensor([params.edge_SOS_token]).to(DEVICE).long()
    
    # init gru_graph hidden state
    gru_graph1.hidden = gru_graph1.init_hidden(num_graphs)
    gru_graph2.hidden = gru_graph2.init_hidden(num_graphs)

    softmax_MHP = nn.Softmax(dim=1)
    softmax = nn.Softmax(dim=0)
    for i in range(params.max_num_node-2):

        # get last generated nodes/edges
        Xsample =  X[:, i]
        Fto_vec = Fto[:,:,i]
        Ffrom_vec = Ffrom[:,:,i]

        # input for gru_graph
        Xsample1  = torch.unsqueeze(X[:,i] ,1)
        Fto_vec  = torch.unsqueeze(Fto_vec, 1)
        Ffrom_vec  = torch.unsqueeze(Ffrom_vec, 1)
        Xsample1 = node_emb(Xsample1)
        Fto_vec = edge_emb(Fto_vec)
        Fto_vec = Fto_vec.contiguous().view(Fto_vec.shape[0],
                                            Fto_vec.shape[1], -1)
        Ffrom_vec = edge_emb(Ffrom_vec)
        Ffrom_vec = Ffrom_vec.contiguous().view(Ffrom_vec.shape[0],
                                                Ffrom_vec.shape[1], -1)
        gru_graph_in = torch.cat((Xsample1.float(), Fto_vec.float(), Ffrom_vec.float()), 2)

        # run one step of gru_graph
        gru_edge_hidden1 = gru_graph1(gru_graph_in, list(np.ones(num_graphs))).data
        gru_edge_hidden2 = gru_graph2(gru_graph_in, list(np.ones(num_graphs))).data

        # exit if EOS token is encountered
        if X[:,i].data.cpu().numpy() == 0:
            X = X[:,0:i]
            Fto = Fto[:,0:i,0:i]
            Ffrom = Ffrom[:,0:i,0:i]
            X_gen = torch.squeeze(X).cpu().numpy()
            Fto_gen = torch.squeeze(Fto).cpu().numpy()
            Ffrom_gen = torch.squeeze(Ffrom).cpu().numpy()
            scene_graph = X_gen, Fto_gen, Ffrom_gen
            break

        if params.egru_num_layers>1:
            gru_edge_hidden1 = torch.cat((gru_edge_hidden1,
                                        torch.zeros(params.egru_num_layers-1,
                                                    gru_edge_hidden1.shape[1],
                                                    gru_edge_hidden1.shape[2]).to(DEVICE)), 0)
            gru_edge_hidden2 = torch.cat((gru_edge_hidden2,
                                        torch.zeros(params.egru_num_layers-1,
                                                    gru_edge_hidden2.shape[1],
                                                    gru_edge_hidden2.shape[2]).to(DEVICE)), 0)
        gru_edge1.hidden = gru_edge_hidden1
        gru_edge2.hidden = gru_edge_hidden2

        # init edge vectors
        Fto_vec = init_edges.clone()
        Ffrom_vec = init_edges.clone()
        for j in range(i+1):
            # input for gru_in
            x1=X[:,j]
            # TODO check the x2 again
            x2=X[:, i+1] 
            fto = Fto_vec[:, j-1] if j>0 else edge_SOS_token
            ffrom = Ffrom_vec[:, j-1] if j>0 else edge_SOS_token
            #print('Inputs to egru', x1, x2, fto, ffrom)
            x1=node_emb(x1.view(x1.shape[0], 1))
            x2=node_emb(x2.view(x2.shape[0], 1))
            fto = edge_emb(fto.view(fto.shape[0], 1))
            ffrom = edge_emb(ffrom.view(ffrom.shape[0], 1))

            if edge_supervision:
                gru_edge_in = torch.cat((x1, x2, fto, ffrom), 2)
            else:
                gru_edge_in = torch.cat((fto, ffrom), 2)
            # run gru_edge and sample next edge
            Ffrom_scores = gru_edge1(gru_edge_in)
            Fto_scores = gru_edge2(gru_edge_in)
            Fto_scores = torch.squeeze(Fto_scores)
            Ffrom_scores = torch.squeeze(Ffrom_scores)
            if use_argmax:
                Fto_sample = torch.argmax(Fto_scores, dim=1)
                Ffrom_sample = torch.argmax(Ffrom_scores, dim=1)
            else:
                if use_MHP:
                    to_rand_idx = int(np.random.choice(params.num_generators, 1))
                    Fto_sample = torch.multinomial(softmax_MHP(Fto_scores.permute(1,0)), 1)[to_rand_idx]
                    from_rand_idx = int(np.random.choice(params.num_generators, 1))
                    Ffrom_sample = torch.multinomial(softmax_MHP(Ffrom_scores.permute(1,0)), 1)[from_rand_idx]
                else:
                    Fto_sample = torch.multinomial(softmax(Fto_scores), 1)
                    Ffrom_sample = torch.multinomial(softmax(Ffrom_scores), 1)

            # update hidden state of gru_edge
            gru_edge1.hidden = gru_edge1.hidden.data.to(DEVICE)
            gru_edge2.hidden = gru_edge2.hidden.data.to(DEVICE)
            # update edge history
            Fto_vec[:, j] = torch.squeeze(Fto_sample)+1
            Ffrom_vec[:, j] = torch.squeeze(Ffrom_sample)+1
        
        # update hidden state of gru_graph
        gru_graph1.hidden = gru_graph1.hidden.data.to(DEVICE)
        gru_graph2.hidden = gru_graph2.hidden.data.to(DEVICE)
        Fto[:,:,i+1] = Fto_vec
        Ffrom[:,:,i+1] = Ffrom_vec
    
    #print(scene_graph)
    return scene_graph



def generate_scene_graphs_given_nodes(main_path, params, config, seed_graphs, models, num_graphs, ind_to_classes, ind_to_predicates, make_visuals=True) :
    graph_data = []
    ordering, weighted_loss, node_pred, edge_pred, use_argmax, use_MHP = get_config(config)
    for idx, sample in enumerate(seed_graphs):
        print(idx)
        category, count = sample
        category = category.detach().numpy()
        count = count.detach().numpy()
        Xin = []
        for cat, c in zip(category, count):
            for _ in range(int(c)):
                Xin.append(int(cat))
        shuffle(Xin)
        Xin = np.array(Xin)
        if Xin.shape[0]<params.max_num_node:
            for idx1 in range(num_graphs):
                scene_graph = generate_edges_given_nodes(params, config, Xin, models)
                if scene_graph is not None:
                    X, Fto, Ffrom = scene_graph
                    F = Fto + np.transpose(Ffrom)
                    graph_data.append([X, F])
                    if make_visuals:
                        sg = vis_graphs(params, X, Fto, Ffrom, ind_to_classes, ind_to_predicates)
                        sg.render(os.path.join(main_path, str(idx)+str(idx1)), view=False)
    return graph_data



# ------------- Conditional node generation -------------------------------------
def generate_nodes_given_edges(params, config, F, models, first_node):
    scene_graph = None
    num_graphs=1
    ordering, weighted_loss, schedule, emb, node_pred, edge_pred, one_ggru, one_egru, use_argmax = get_config(config)
    node_emb, edge_emb, mlp_node, gru_graph1, gru_graph2, gru_graph3, gru_edge1, gru_edge2 = models
    # set models to eval mode
    if emb:
        node_emb.eval()
        edge_emb.eval()
    mlp_node.eval()
    gru_graph3.eval()

    # intantiate generated graphs
    X = torch.zeros(num_graphs,
                    params.max_num_node-1).to(DEVICE).long()
    Fto = torch.zeros(num_graphs,
                      params.max_num_node-1,
                      params.max_num_node).to(DEVICE).long()
    Ffrom = torch.zeros(num_graphs,
                      params.max_num_node-1,
                      params.max_num_node).to(DEVICE).long()
    Fto_given = np.triu(F, +1)
    Ffrom_given = np.transpose(np.tril(F, -1))
    l = F.shape[0]
    Fto[0,:l,:l] = torch.Tensor(Fto_given).to(DEVICE).long()
    Ffrom[0,:l,:l] = torch.Tensor(Ffrom_given).to(DEVICE).long()

    # sample initial object: assume equal probability of each object category
    Xsample = torch.Tensor([first_node-1]).long().to(DEVICE)

    # initial edge vector
    init_edges = torch.zeros(num_graphs, params.max_num_node-1).to(DEVICE).long()
    edge_SOS_token = torch.Tensor([params.edge_SOS_token]).to(DEVICE).long()
    
    # init gru_graph hidden state
    gru_graph3.hidden = gru_graph3.init_hidden(num_graphs)
    
    for i in range(params.max_num_node-1):
        # update graph info with generated nodes/edges
        X[:,i] = Xsample+1
        Fto_vec = Fto[:,:,i]
        Ffrom_vec = Ffrom[:,:,i]
        # print('node: ', i, X[:,i])

        # input for gru_graph
        # if not only_edge:
        #    Xsample1  = torch.unsqueeze(Xsample, 1)
        # else:
        Xsample1  = torch.unsqueeze(X[:,i] ,1)
        Fto_vec  = torch.unsqueeze(Fto_vec, 1)
        Ffrom_vec  = torch.unsqueeze(Ffrom_vec, 1)
        # print(Xsample1.shape, Fto_vec.shape, Ffrom_vec.shape)
        if emb:
            Xsample1 = node_emb(Xsample1)
            Fto_vec = edge_emb(Fto_vec)
            Fto_vec = Fto_vec.contiguous().view(Fto_vec.shape[0],
                                                Fto_vec.shape[1], -1)
            Ffrom_vec = edge_emb(Ffrom_vec)
            Ffrom_vec = Ffrom_vec.contiguous().view(Ffrom_vec.shape[0],
                                                    Ffrom_vec.shape[1], -1)
        gru_graph_in = torch.cat((Xsample1.float(), Fto_vec.float(), Ffrom_vec.float()), 2)
        
        mlp_input = gru_graph3(gru_graph_in, list(np.ones(num_graphs))).data
        # run mlp_node and sample next object
        Xscores = mlp_node(mlp_input)
        # print('Xscores', Xscores.shape, Xscores)
        # Xscores = torch.squeeze(Xscores,0)
        if use_argmax:
            Xsample = torch.argmax(Xscores, dim=2)
            # print('next node', Xsample, Xsample.shape)
        else:
            Xscores = torch.squeeze(Xscores)
            Xsample = torch.multinomial(Xscores, 1)
            # print('next node', Xsample, Xsample.shape)

        # exit if EOS token is encountered
        if Xsample.data.cpu().numpy()==params.node_EOS_token or i==params.max_num_node-1:
            if i==0:
                break
            else:
                X = X[:,0:i+1]
                Fto = Fto[:,0:i+1,0:i+1]
                Ffrom = Ffrom[:,0:i+1,0:i+1]
                X_gen = torch.squeeze(X).cpu().numpy()
                Fto_gen = torch.squeeze(Fto).cpu().numpy()
                Ffrom_gen = torch.squeeze(Ffrom).cpu().numpy()
                scene_graph = X_gen, Fto_gen, Ffrom_gen
                break
    
    return scene_graph

        
        
def generate_scene_graphs_given_edges(main_path, params, config, seed_graphs, models, num_graphs, ind_to_classes, ind_to_predicates):
    print('Generating samples..')
    ordering, weighted_loss, schedule, emb, node_pred, edge_pred, one_ggru, one_egru, use_argmax = get_config(config)
    for idx, sample in enumerate(seed_graphs):
        Xin = sample[0]
        Fin = sample[1]
        if ordering=='random':
            Xin, Fin = random_ordered(Xin, Fin)
        elif ordering =='bfs':
            root =  random.choice(np.arange(Xin.shape[0]))         
            Xin, Fin = bfs_ordered(Xin, Fin, root)
        elif ordering == 'predefined':
            Xin, Fin = predefined_ordered(Xin, Fin)
        elif ordering == 'none':
            pass
        elif ordering=='3dbfs':
            root = list(Xin).index(32)
            Xin, Fin = bfs_ordered(Xin, Fin, root)
        else:
            raise print('Ordering not understood')
        
        Fto = np.triu(Fin, +1)
        Ffrom = np.transpose(np.tril(Fin, -1))

        if Xin.shape[0]<=params.max_num_node:
            path = os.path.join(main_path, str(idx))
            os.makedirs(path, exist_ok=True)
        
            sg = Digraph('sg', format='png')
            for idx, node in enumerate(Xin-1):
            #for idx, node in enumerate(Xin):
                sg.node(str(idx), ind_to_classes[node])
            to_edge_obj, to_edge_subj = np.nonzero(Fto)
            for obj, subj in zip(to_edge_obj, to_edge_subj):
                sg.edge(str(obj), str(subj), label=ind_to_predicates[int(Fto[obj, subj]-1)])
                #sg.edge(str(obj), str(subj), label=ind_to_predicates[int(Fto[obj, subj])])
            from_edge_subj, from_edge_obj = np.nonzero(Ffrom)
            for obj, subj in zip(from_edge_obj, from_edge_subj):
                sg.edge(str(obj), str(subj), label=ind_to_predicates[int(Ffrom[subj, obj]-1)])
                #sg.edge(str(obj), str(subj), label=ind_to_predicates[int(Ffrom[subj, obj])])
            sg.render(os.path.join(path, 'Ground_truth'), view=False)

            for idx in range(num_graphs):
                graph = generate_nodes_given_edges(params, config, Fin, models, Xin[0])
                print(graph)
                if graph is not None:
                    X, Fto, Ffrom = graph
                    #print(X, Fto, Ffrom)
                    sg = vis_graphs(params, X, Fto, Ffrom, ind_to_classes, ind_to_predicates)
                    sg.render(os.path.join(path, str(idx)), view=False)


def generate_nodes(params, config, F, models, first_node):
    X_gen = None
    num_graphs=1
    ordering, weighted_loss, schedule, emb, node_pred, edge_pred, one_ggru, one_egru, use_argmax = get_config(config)
    node_emb, edge_emb, mlp_node, gru_graph1, gru_graph2, gru_graph3, gru_edge1, gru_edge2 = models
    # set models to eval mode
    if emb:
        node_emb.eval()
        edge_emb.eval()
    mlp_node.eval()
    gru_graph3.eval()

    # intantiate generated graphs
    X = torch.zeros(num_graphs,
                    params.max_num_node-1).to(DEVICE).long()
    Fto = torch.zeros(num_graphs,
                      params.max_num_node-1,
                      params.max_num_node).to(DEVICE).long()
    Ffrom = torch.zeros(num_graphs,
                      params.max_num_node-1,
                      params.max_num_node).to(DEVICE).long()
    # Fto_given = np.triu(F, +1)
    # Ffrom_given = np.transpose(np.tril(F, -1))
    # l = F.shape[0]
    # Fto[0,:l,:l] = torch.Tensor(Fto_given).to(DEVICE).long()
    # Ffrom[0,:l,:l] = torch.Tensor(Ffrom_given).to(DEVICE).long()

    # sample initial object: assume equal probability of each object category
    Xsample = torch.Tensor([first_node-1]).long().to(DEVICE)

    # initial edge vector
    init_edges = torch.zeros(num_graphs, params.max_num_node-1).to(DEVICE).long()
    edge_SOS_token = torch.Tensor([params.edge_SOS_token]).to(DEVICE).long()
    
    # init gru_graph hidden state
    gru_graph3.hidden = gru_graph3.init_hidden(num_graphs)
    
    for i in range(params.max_num_node-1):
        # update graph info with generated nodes/edges
        X[:,i] = Xsample+1
        #Fto_vec = Fto[:,:,i]
        #Ffrom_vec = Ffrom[:,:,i]
        #print('node: ', i, X[:,i])

        # input for gru_graph
        # if not only_edge:
        #    Xsample1  = torch.unsqueeze(Xsample, 1)
        # else:
        Xsample1  = torch.unsqueeze(X[:,i] ,1)
        # Fto_vec  = torch.unsqueeze(Fto_vec, 1)
        # Ffrom_vec  = torch.unsqueeze(Ffrom_vec, 1)
        # print(Xsample1.shape, Fto_vec.shape, Ffrom_vec.shape)
        if emb:
            Xsample1 = node_emb(Xsample1)
            # Fto_vec = edge_emb(Fto_vec)
            # Fto_vec = Fto_vec.contiguous().view(Fto_vec.shape[0],
            #                                     Fto_vec.shape[1], -1)
            # Ffrom_vec = edge_emb(Ffrom_vec)
            # Ffrom_vec = Ffrom_vec.contiguous().view(Ffrom_vec.shape[0],
            #                                         Ffrom_vec.shape[1], -1)
        # gru_graph_in = torch.cat((Xsample1.float(), Fto_vec.float(), Ffrom_vec.float()), 2)
        
        mlp_input = gru_graph3(Xsample1, list(np.ones(num_graphs))).data
        # run mlp_node and sample next object
        Xscores = mlp_node(mlp_input)
        # print('Xscores', Xscores.shape, Xscores)
        # Xscores = torch.squeeze(Xscores,0)
        if use_argmax:
            Xsample = torch.argmax(Xscores, dim=2)
            #print('next node', Xsample, Xsample.shape)
        else:
            Xscores = torch.squeeze(Xscores)
            Xsample = torch.multinomial(Xscores, 1)
            #print('next node', Xsample, Xsample.shape)

        # exit if EOS token is encountered
        if Xsample.data.cpu().numpy()==params.node_EOS_token or i==params.max_num_node-1:
            if i==0:
                break
            else:
                X = X[:,0:i+1]
                X_gen = torch.squeeze(X).cpu().numpy()
                break
    #print(X_gen)
    return X_gen





def generate_nodes_sg(main_path, params, config, seed_graphs, models, num_graphs, ind_to_classes, ind_to_predicates,make_visuals=False):
    print('Generating samples..')
    ordering, weighted_loss, schedule, emb, node_pred, edge_pred, one_ggru, one_egru, use_argmax = get_config(config)
    node_data = []
    node_freq = dict()
    for idx, sample in enumerate(seed_graphs):
        Xin = sample[0]
        Fin = sample[1]
        if ordering=='random':
            Xin, Fin = random_ordered(Xin, Fin)
        elif ordering =='bfs':
            root =  random.choice(np.arange(Xin.shape[0]))         
            Xin, Fin = bfs_ordered(Xin, Fin, root)
        elif ordering == 'predefined':
            Xin, Fin = predefined_ordered(Xin, Fin)
        elif ordering == 'none':
            pass
        elif ordering=='3dbfs':
            root = list(Xin).index(32)
            Xin, Fin = bfs_ordered(Xin, Fin, root)
        else:
            raise print('Ordering not understood')
        
        Xin = np.array(list(set(list(Xin))))
        Xin.sort()
        #print('Ground truth', Xin)
        Fto = np.triu(Fin, +1)
        Ffrom = np.transpose(np.tril(Fin, -1))

        if Xin.shape[0]<=params.max_num_node:
            # path = os.path.join(main_path, str(idx))
            # #os.makedirs(path, exist_ok=True)
            # sg = Digraph('sg', format='png')
            # for idx, node in enumerate(Xin-1):
            #     sg.node(str(idx), ind_to_classes[node])
            # sg.render(os.path.join(path, 'Ground_truth'), view=False)
            for idx1 in range(num_graphs):
                X = generate_nodes(params, config, Fin, models, Xin[0])
                #print(X)
                if X is not None:
                    X = list(set(X))
                    node_data.append(X)
                    for node in X:
                        obj = ind_to_classes[node-1]
                        if obj not in node_freq:
                            node_freq[obj] = 1
                        else:
                            node_freq[obj] += 1
                    if make_visuals:
                        sg = Digraph('sg', format='png')
                        for idx, node in enumerate(X-1):
                            sg.node(str(idx), ind_to_classes[node])
                        path = os.path.join(main_path, str(idx))
                        os.makedirs(path, exist_ok=True)
                        sg.render(os.path.join(path, str(idx1)), view=False)
    node_freq = {k: v for k, v in sorted(node_freq.items(), key=lambda item: item[1], reverse=True)}
    return node_freq



########################### VAE NODE SET GENERATION #######################################################

def generate_objects_VAE(params, models, num_samples):
    """
    Deterministically transform noise to a set of objects
    """
    cardinality_emb, category_emb, count_emb, set_encoder, set_decoder, VAE_model = models
    cardinality_emb.eval()
    category_emb.eval()
    count_emb.eval()
    set_encoder.eval()
    set_decoder.eval()
    VAE_model.eval()
    generated_data = []
    # random sampler
    noise_sampler = torch.distributions.MultivariateNormal(torch.zeros(params.latent_size), torch.eye(params.latent_size))
    z_all = noise_sampler.sample((num_samples,)).to(DEVICE)
    softmax_card=nn.Softmax(dim=1)
    softmax_count=nn.Softmax(dim=0)
    batch = 0
    for batch in range(int(num_samples/params.gen_batch)):
        start_id = batch*params.gen_batch
        end_id = (batch+1)*params.gen_batch-1
        z = z_all[start_id:end_id]
        score_obj_cat = set_decoder.decode_objects(z)
        #score_obj_cat, score_obj_count  = set_decoder(z)
        # generate cardinality
        #cardinality = torch.multinomial(softmax_card(score_card), 1)+1
        #cardinality = torch.argmax(softmax_card(score_card), dim=1)+1
        batch+=1
        idx=0
        for cat_scores in score_obj_cat:
            # generate objects
            #obj_scores, obj = torch.topk(cat_scores, k=int(c), dim=0, largest=True)
            obj = torch.nonzero(torch.bernoulli(cat_scores)).squeeze(dim=1)
            print(cat_scores[obj])
            # generate counts autoregressively
            obj_counts = torch.zeros(obj.shape).to(DEVICE)
            count_lst = torch.zeros(params.category_dim).to(DEVICE)
            for i, o in enumerate(obj):
                count_mlp_input = torch.zeros((1, params.count_mlp_input, 1)).to(DEVICE)
                count_mlp_input[:, obj, :] = 1
                count_mlp_input[:, params.category_dim+o, :] = 1
                count_mlp_input[0, 2*params.category_dim:3*params.category_dim, 0] = count_lst
                set_decoder.count_hidden = z[idx].reshape((1, 1, z[idx].shape[0]))
                #print(set_decoder.decode_counts_autoregressive(z[idx], count_mlp_input))
                count = torch.multinomial(softmax_count(set_decoder.decode_counts_autoregressive(count_mlp_input).squeeze()), 1).to(DEVICE)
                count_lst[o] = count
                obj_counts[i] = count
            idx+=1
            if obj.shape[0]>1:
                #obj_counts =  torch.argmax(softmax_count(count_scores)[:,obj], dim=0) + 1
                #obj_counts =  torch.multinomial(softmax_count(count_scores).permute(1,0)[obj], 1).squeeze(dim=1)+1
                print(obj+1, obj_counts)
                generated_data.append((obj+1, obj_counts))
    return generated_data






######### setVAE + GraphRNN Generation #################################

def generate_VAE_graphRNN(gen_path, params, config, VAEmodel, graphRNNmodel, num_samples, ind_to_classes, ind_to_predicates):
    gen_dataset = []
    nodeset_list = generate_objects_VAE(params, VAEmodel, num_samples)
    for nodeset in nodeset_list:
        Xin = []
        for node, count in nodeset:
            for _ in range(count):
                Xin.append(node)
        obj_idx_pair = [[idx, obj] for idx, obj in enumerate(Xin)]
        sorted_pair = np.array(sorted(obj_idx_pair, key = lambda x: x[1]))
        order_idx = [int(i) for i in sorted_pair[:,0]]
        Xin = Xin[order_idx]
        scene_graph = generate_edges_given_nodes(params, config, Xin, graphRNNmodel)
        gen_dataset.append(scene_graph)

    return gen_dataset