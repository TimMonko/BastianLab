image1 = getTitle();
run("Crop");
run("Duplicate...", " ");
roiManager("Add");

image2 = getTitle();
run("Select All");
run("Gaussian Blur...", "sigma=5");
roiManager("Select", 0);
run("Clear", "slice");
roiManager("Delete");

selectWindow(image1);
run("Clear Outside");

imageCalculator("Add create", image1, image2);
run("Select All");
run("Enhance Contrast", "saturated=0.35");