# setup environment
`conda install --file environment.yml`  
`conda activate ttsr`

# add custom_size function
Default: custom_size == False, which outputs following the original ttsr setting    
To active custom size: `--custom_size=True`, then the default output size is 1920x1080
To custom size: `--image_height=xxx`
                `--image_width=xxx`

# add process visualization function

# bug fixing
minor bug fixed, used not able to test single channel gray scale image, which can do now