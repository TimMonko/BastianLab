run("HiLo");
run("Properties...", "unit=µm pixel_width=.064 pixel_height=.064");
run("Deinterleave", "how=2 keep");
run("Z Project...", "projection=[Max Intensity]");
run("Enhance Contrast", "saturated=0.35");

//run("Subtract Background...", "rolling=30 sliding");
run("Convoluted Background Subtraction", "convolution=Median radius=30");
run("Top Hat...", "radius=5");
run("Median...", "radius=2");
run("Extended Min & Max", "operation=[Extended Minima] dynamic=30 connectivity=4");

