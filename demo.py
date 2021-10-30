
import os

# Clone GFPGAN and enter the GFPGAN folder

os.system("git clone https://github.com/TencentARC/GFPGAN.git")  
os.system("cd GFPGAN")  
os.system("pip install -r requirements.txt")  
os.system("pip install basicsr")  
os.system("pip install gradio")  
os.system("python setup.py develop")  
os.system("pip install realesrgan")  
os.system("wget https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth -P experiments/pretrained_models")  

import gradio as gr
import torch
from PIL import Image
import glob
import random


def clear_dir(path):
  files = glob.glob(f"{path}*")
  for f in files:
      os.remove(f)

def locate(filename):
  paths = ['results/restored_faces/' ,'results/restored_imgs/']
  for path in paths:
    for trial in os.listdir(path):
      if filename in trial:
        return path+trial

# Define the main function
def generate(img):
    input_path = 'inputs/upload/'
    path_results = 'results/restored_faces/'
    clear_dir(input_path)
    clear_dir(path_results)

    inp = Image.fromarray(img.astype('uint8'), 'RGB')

    width, height = inp.size

    while width > 256 or height > 256:
      inp = inp.resize((width//2 , height//2))
      width, height = inp.size


    
    r = random.randint(0,100000)
    inp.save(f"{input_path}{r}.jpg") 

    os.system("python inference_gfpgan.py --upscale 2 --test_path inputs/upload --save_root results --model_path experiments/pretrained_models/GFPGANCleanv1-NoCE-C2.pth --bg_upsampler realesrgan")  

    files = os.listdir(path_results)

    return locate(str(r))


img = gr.inputs.Image()
# Launch the de
iface = gr.Interface(generate, inputs=[img], server_port = 21278, article = """
     Supplementary {
       -  Please contact me on golan.itamar@gmail.com
       -  Follow me on Linkedin for similar projects - https://www.linkedin.com/in/itamar-g1/
       -  Execution time takes about ~60 seconds
    }
""",
                                        
                     outputs=['image'], enable_queue=True, live=False , verbose = True ,allow_screenshot=False, allow_flagging=False, title = "Free Image Restoration using GFPGAN",
css = """

# * {
# 	margin: 0;
# 	padding: 0;
# 	box-sizing: border-box;
# 	}

# html {
#   font-color : white;
# 	height: 2000px;
# 	background-color: red;
# 	background-image: url(https://upload.wikimedia.org/wikipedia/commons/6/64/Creaci%C3%B3n_de_Ad%C3%A1n_%28Miguel_%C3%81ngel%29.jpg);
# 	background-size: 100% auto;
# 	background-position: top center;
# 	background-attachment: fixed;
# 	}


# body {
# 	height: 100%;
# 	background-size: 50% 50%;
# 	mix-blend-mode: luminosity;
# 	}

@import url(https://fonts.googleapis.com/css?family=Fjalla+One);
body{
	background-image: url(https://upload.wikimedia.org/wikipedia/commons/6/64/Creaci%C3%B3n_de_Ad%C3%A1n_%28Miguel_%C3%81ngel%29.jpg);
}
html {
  font-family: 'Fjalla One', sans-serif;
  color: firebrick;
  width: 100%;
  height: 150%;
  background: rgb(169, 3, 41);

  overflow: hidden;
  position: bottom
  # background: rgb(169, 3, 41);
  /* Old browsers */
  /* IE9 SVG, needs conditional override of 'filter' to 'none' */
  # background: radial-gradient(ellipse at center, rgba(169, 3, 41, 1) 0%, rgba(143, 2, 34, 1) 44%, rgba(109, 0, 25, 1) 100%);
}

.rainbow-text {
  position: absolute;
  top: 50%;
  color : blue

  display: block;
  left: 50%;
  margin-left: -99px;
  margin-top: -100px;
}

.title-text {
  text-align: center;
}

.title-text span.letters {
  font-size: 6rem;
}

.letters {
  color: #FFF;
  text-shadow: 2px 2px 6px #333;
  animation: color-changer 8s infinite;
  transition: color-changer ease-in-out;
  @for $i from 1 through 7 {
    &:nth-of-type(#{$i}) {
      animation-delay: #{$i * 100}ms;
    }
  }
}


""")
iface.launch(debug=True)