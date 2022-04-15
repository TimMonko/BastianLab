///TOPHAT WITH FIND MAXIMA THRESHOLDING
idOrig = getImageID();
selectImage(idOrig);
run("Convoluted Background Subtraction", "convolution=Median radius=10");
//AN ELEMENT OF TWO, ESPECIALLY FOR LAPLACIAN IS MUCH LESS NOISY, BUT DEFINITELY CAUSES GREAT BLURRING
//run("Morphological Filters", "operation=Closing element=Disk radius=1");
//run("Median...", "radius=0.5");
run("FeatureJ Laplacian", "compute smoothing=1");
run("HiLo");
run("Invert LUT");
idLap = getImageID();
//MAXIMA POINTS
selectImage(idLap);
run("Duplicate...", "title=[lap-maxima]");
run("Find Maxima...", "prominence=1000 light output=[Single Points]");
idLapMax = getImageID();
//MAXIMA WATERSHED
selectImage(idLap);
run("Duplicate...", "title=[lap-watershed]");
run("Find Maxima...", "prominence=1000 light output=[Segmented Particles]");
idLapWatershed = getImageID();

//THRESHOLD MASK
selectImage(idLap);
run("Duplicate...", "title=[lap-thresh]");
run("Median...", "radius=0.5");
setAutoThreshold("Otsu");
setOption("BlackBackground", false);
run("Convert to Mask");
idLapThresh = getImageID();
imageCalculator("AND create", "lap-thresh","lap-watershed Segmented");



run("Binary Feature Extractor", "objects=[Result of lap-thresh] selector=[lap-maxima Maxima] object_overlap=1 combine count");
//imageCalculator("AND create", "Extracted_lap-thresh-1","lap-watershed Segmented");

run("Invert LUT");
run("Watershed");
run("Analyze Particles...", "  circularity=0.5-1.00 show=Masks clear add");