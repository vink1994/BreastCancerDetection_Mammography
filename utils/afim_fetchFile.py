
def afim_fetchFile(path):
    with open(path, 'r') as f:
        r = f.read()
        r = r.replace('=', '+').replace('\n', '+').split('+')
        new_r = []
        for i in r:
            if i=='True':
                new_r.append('1')
            elif i == 'False':
                new_r.append(0)
            elif i !='' and '#' not in i:
                new_r.append(i)
        print(new_r)
    return new_r
print("loading AFIM Configuration")   
afim_fetchFile('G:/Sharda_Code/To send/V6/AFIM/afim_conf_mod/AFIM_configm.txt')    