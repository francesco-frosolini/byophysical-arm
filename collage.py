from PIL import Image, ImageOps

def create_collage(image_paths, output_path):
    # Constants
    COLS = 2
    ROWS = 3
    IMG_W, IMG_H = 800, 600
    BORDER = 4
    
    # Calculate canvas size
    # (Width * cols) + borders between and on edges
    canvas_w = (IMG_W * COLS) + (BORDER * (COLS + 1))
    canvas_h = (IMG_H * ROWS) + (BORDER * (ROWS + 1))
    
    # Create white background (change (0,0,0) for black border)
    collage = Image.new('RGB', (canvas_w, canvas_h), color=(0, 0, 0))
    
    for index, path in enumerate(image_paths):
        if index >= COLS * ROWS:
            break
            
        # Open and ensure correct size
        img = Image.open(path).convert("RGB")
        img = img.resize((IMG_W, IMG_H))
        
        # Calculate grid position
        column = index % COLS
        row = index // COLS
        
        # Calculate X, Y coordinates
        x = (column * IMG_W) + (BORDER * (column + 1))
        y = (row * IMG_H) + (BORDER * (row + 1))
        
        collage.paste(img, (x, y))
    
    collage.save(output_path)
    print(f"Collage saved to {output_path}")

# List of 6 image filenames
folder = "muscle_torque_map/90def/"
images = [f"plots/{folder}{i}.png" for i in range(6)] 
create_collage(images, f"plots/{folder}collage.png")