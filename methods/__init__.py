import os
import importlib

def get_all_methods():
    return [method.split('.')[0] for method in os.listdir('methods')
            if not method.find('__') > -1 and 'py' in method]

names = {}
for method in get_all_methods():
    mod = importlib.import_module('methods.' + method)
    class_name = {x.lower():x for x in mod.__dir__()}[method.lower()]
    names[method] = getattr(mod, class_name)

def get_method(nets_list,args, transform):
    return names[args.method](nets_list,args,transform)
