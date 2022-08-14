from copy import copy





def get_nearest_resource(data, columns, corr = True, work_map=None):
    """Function for getting an ordered list of resources by distance metric
    Args:
        data (DataFrame): matrix of distances between works and resources
        columns (list): list of works
        corr (bool, optional): distance type flag. If True, then the distance is according to the type of correlation, 
        that is, the more, the closer the objects. If False, then the distance is by type of distance, that is, the smaller, the closer the objects.
        Defaults to True.
        work_map (dict, optional): dictionary of resource sheets. Defaults to None.

    Returns:
        list, list: ordered list of resources, list of distance values
    """    
    groups = []
    group_values = []
    if work_map == None:
        if corr:
            for c in columns:
                close_ind = data[c + '_act_fact'].sort_values(ascending=False).index
                vals = data[c + '_act_fact'].sort_values(ascending=False).values
                close_res = []
                close_values = []
                for i, element in enumerate(close_ind):
                    if element.split('_')[-2] == 'res':
                        close_res.append(element)
                        close_values.append(vals[i])
                groups.append(close_res)
                group_values.append(close_values)
        else:
            for c in columns:
                close_ind = data[c + '_act_fact'].sort_values().index
                vals = data[c + '_act_fact'].sort_values().values
                close_res = []
                close_values = []
                for i, element in enumerate(close_ind):
                    if element.split('_')[-2] == 'res':
                        close_res.append(element)
                        close_values.append(vals[i])
                groups.append(close_res)
                group_values.append(close_values)
    else:
        if corr:
            for c in columns:
                new_data = copy(data)
                for i in new_data.index:
                    if (i.split('_')[-2] == 'res') & (i in work_map[c+ '_act_fact']):
                        new_data.loc[i, 'map'] = 1
                    else:
                        new_data.loc[i, 'map'] = 0
                close_ind = new_data.sort_values(by=[c + '_act_fact', 'map'], ascending=False).index
                vals = new_data.sort_values(by=[c + '_act_fact', 'map'], ascending=False)[c + '_act_fact'].values
                close_res = []
                close_values = []
                for i, element in enumerate(close_ind):
                    if element.split('_')[-2] == 'res':
                        close_res.append(element)
                        close_values.append(vals[i])
                groups.append(close_res)
                group_values.append(close_values)
        else:
            for c in columns:
                new_data = copy(data)
                for i in new_data.index:
                    if (i.split('_')[-2] == 'res') & (i in work_map[c+ '_act_fact']):
                        new_data.loc[i, 'map'] = 0
                    else:
                        new_data.loc[i, 'map'] = 1
                close_ind = new_data.sort_values(by=[c + '_act_fact', 'map']).index
                vals = new_data.sort_values(by=[c + '_act_fact', 'map'])[c + '_act_fact'].values
                close_res = []
                close_values = []
                for i, element in enumerate(close_ind):
                    if element.split('_')[-2] == 'res':
                        close_res.append(element)
                        close_values.append(vals[i])
                groups.append(close_res)
                group_values.append(close_values)
    return groups, group_values