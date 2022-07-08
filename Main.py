import argparse
import random
import torchvision.transforms as transforms

from input_control_methods import *
from style_transfer import *
from train import train_model

parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

parser.add_argument('--unzip', type=str, default='False')

parser.add_argument('--train', type=str, default='False',
                       choices=['Shape','Style'])

parser.add_argument('--generation_type', type=str, default='Full',
                       choices=['Full', 'Shape', 'Style'])

parser.add_argument('--shape_generation', type=str, default='randomReconstruction',
                       choices=['randomReconstruction', 'selectedReconstruction', 'randomSample'])

parser.add_argument('--shape_file', type=str, default='1114700_0.jpg')
    
parser.add_argument('--gender', type=str, default='Mens',
                   choices =['Womens','Mens','Unisex']) 
    
parser.add_argument('--sleeve_length', type=str, default=None,
                        choices=['ShortSleeves', 'LongSleeves', 'Sleeveless','TankTops'])
    
parser.add_argument('--garment_type', type=str, default=None, 
                        choices=['ButtonUpShirts','Buttondown','Crew','Pullover','Vneck','Blouses','Raglan','Workout','Athletic','TShirts','Polos'])
    
parser.add_argument('--style_images', type=str, default='Random', choices=['Random', 'Selected'])

parser.add_argument('--style1', type=str, default='abstract_brown (5).jpg')

parser.add_argument('--style2', type=str, default='abstract_brown (39).jpg')
 
parser.add_argument('--interpolation_steps', type=int, default=5)
    
    
args = parser.parse_args()
print(args)

if args.unzip == 'True':
    unzip_datasets()
    
if args.train == 'Shape':
    train_model('Shape', batch_size=16, img_size=128, path='output/', opt='SGD', lr=0.01, momentum=0.95, factor=1)
    
elif args.train == 'Style':
    train_model('Style', batch_size=16, img_size=128, path='output/', opt='SGD', lr=0.01, momentum=0.95, factor=0.1)
    

if args.train == 'False' and args.unzip == 'False':
    #loads the current pretrained models for evaluation
    shape_model, style_model = load_current_models()
    shape_model.eval()
    style_model.eval()
    

    #Shape generation process, either sampling an image or using a selected or random reconstruction
    if args.generation_type == 'Full' or args.generation_type == 'Shape':
        if args.shape_generation == 'randomSample':
            print("Random shape sample: ")
            shape, shapez = sample_generic(shape_model)

        elif args.shape_generation == 'randomReconstruction':
            print("Reconstruct:")
            shape, shapez = reconstruct(shape_model, random.choice(os.listdir("./data/apply_data_shapes/")))
            show_images(shape, 'randomReconstruction')

        elif args.shape_generation == 'selectedReconstruction':
            print("Reconstruct:")
            shape, shapez = reconstruct(shape_model, args.shape_file)
            show_images(shape, 'selectedReconstruction')

        else: 
            print("No shape generation type chosen, please choose one to run the program successfully")
            shapez = None

        att1 = generate_attribute_vector(shape_model, gender=args.gender, attribute=args.sleeve_length, attribute2=args.garment_type)

        if att1 is not None:
            print("applied attributes:")
            applied, appliedz = apply_attribute(shape_model, shapez, att1, num_steps=args.interpolation_steps, show=False)
            show_images(applied, "appliedAttribute_interpolation_{}_{}.jpg".format(args.sleeve_length, args.garment_type))
                
            for k, img in enumerate(applied):
                img = img.cpu().detach().numpy().transpose(1, 2, 0) * 255
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cv2.imwrite('./output/attributes_{}_{}_{}.jpg'.format(args.sleeve_length, args.garment_type, k),img)
                
            shapez = appliedz[-1]
        
    if args.generation_type == 'Full' or args.generation_type == 'Style':
        if args.style_images == 'Selected':
            patt1, pattz1= decode_pattern(style_model, args.style1)
            show_images(patt1,'style1')
            patt2, pattz2= decode_pattern(style_model, args.style2)
            show_images(patt2,'style2')
        elif args.style_images == 'Random':
            patt1, pattz1= decode_pattern(style_model, random.choice(os.listdir("./data/apply_data_styles/")))
            show_images(patt1,'randomPattern1')
            patt2, pattz2= decode_pattern(style_model, random.choice(os.listdir("./data/apply_data_styles/")))
            show_images(patt2,'randomPattern2')

        interpolated_patt1, interpolated_pattz1, slerp_patt1, slerp_patt_z1 = interpolation(pattz1, pattz2, args.interpolation_steps, style_model, title="styleInterpolation")

    if args.generation_type == 'Full':
        style_int = apply_style_interpolation(shape_model, style_model, slerp_patt_z1, shapez)
                  
        show_images(style_int, "styleInterpolation")

        
print("Done")
