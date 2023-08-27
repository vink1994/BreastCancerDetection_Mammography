from AFIM_My_model import afim_enh_mod, afim_basemod_des

def AFIM_Init_mod(str_model, n, num_classes=1, weights=None, hyper_param3=False, hyper_param2=True, visualize=False):
   
    print('Developed Model for CBIS Dataset:', str_model)
    print("choosing the best model")
    
    if str_model == 'afim_base_mod':
        return afim_basemod_des.AFIMbasenetmod1(num_classes, channels=n, visualize=visualize)
    elif str_model == 'AFIMDeepNetmod1':
        return afim_enh_mod.AFIMDeepNetmod1(channels=2, n=n, num_classes=num_classes, visualize=visualize)
    if str_model == 'AFIMbasenetmod2':
        return afim_basemod_des.AFIMbasenetmod2(num_classes, channels=n)
    elif str_model == 'AFIMDeepNetmod2':
        return afim_enh_mod.AFIMDeepNetmod2(channels=2, n=n, num_classes=num_classes)
    if str_model == 'AFIMDeepModNet3':
        return afim_basemod_des.AFIMDeepModNet4(hyper_param3=hyper_param3, num_classes=num_classes, weights=weights)
    elif str_model == 'AFIMDeepEncModNet3':
        return afim_enh_mod.AFIMDeepEncModNet3(n=n, hyper_param3=hyper_param3, num_classes=num_classes, weights=weights)

    if str_model == 'AFIMDeepModNet4':
        return afim_basemod_des.AFIMDeepModNet4(num_classes=num_classes, weights=weights, hyper_param2=hyper_param2, visualize=visualize)
    elif str_model == 'AFIMDeepEncModNet4':
        return afim_enh_mod.AFIMDeepEncModNet4(n=n, num_classes=num_classes, weights=weights, hyper_param2=hyper_param2, visualize=visualize)

    else:
        raise ValueError ('Check the model you typed.')
        
        
