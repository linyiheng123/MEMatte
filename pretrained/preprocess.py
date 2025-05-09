import torch
def add_teacher(model, name):
    new_model = {}
    for k in model.keys():
        new_model[k] = model[k]
        if 'backbone' in k:
            new_model['teacher_'+k] = model[k]
            print(f'teacher_{k}')

    torch.save(new_model, name + '.pth')

if __name__ == "__main__":
    # Downloading the official checkpoint of ViTMatte, and then process the checkpoint with the script. 
    # ViTMatte_S_Com.pth: https://drive.google.com/file/d/12VKhSwE_miF9lWQQCgK7mv83rJIls3Xe/view
    # ViTMatte_B_Com.pth: https://drive.google.com/file/d/1mOO5MMU4kwhNX96AlfpwjAoMM4V5w3k-/view?pli=1
    teacher_model = torch.load('ViTMatte_S_Com.pth')['model']
    add_teacher(teacher_model, 'ViTMatte_S_Com_with_teacher')