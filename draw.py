def draw_nx_graph(nx_graph,output_file):
    """ 
        Generates graphs (png image) from networkx graph object

    Args:
        nx_graph(str): networkx graph object to be drawn
        output_drectory(str): location to save graph
        output_file_name(str): name of the image file to be saved

    Returns:
       null

    """
    for u, v, d in nx_graph.edges(data=True):
        u = u.name +"  " +str(u.bz_critical)
        v= v.name +"  " +str(v.bz_critical)
        d['label'] = d.get('weight', '')

    A = nx.nx.drawing.nx_agraph.to_agraph(nx_graph)
    A.layout(prog='dot')
    A.draw(output_file)