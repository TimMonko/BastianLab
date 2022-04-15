// Prompt to get sigma values for the Difference of Gaussians filtering
Dialog.create("Choose filter sizes for DoG filtering");
Dialog.addNumber("Gaussian sigma 1", 1);
Dialog.addNumber("Gaussian sigma 2", 2);
Dialog.show();
sigma1 = Dialog.getNumber();
sigma2 = Dialog.getNumber();

// Get the current image
idOrig = getImageID();



/////DIFFERENCE OF GAUSSIAN
run("Duplicate...", "title=[sigma1]");
// Convert to 32-bit (so we can have negative values)
run("32-bit");
idSigma1 = getImageID();
run("Gaussian Blur...", "sigma="+sigma1);

selectImage(idOrig);
run("Duplicate...", "title=[sigma2]");
// Convert to 32-bit (so we can have negative values)
run("32-bit");
idSigma2 = getImageID();
run("Gaussian Blur...", "sigma="+sigma2);
// Subtract one smoothed image from the other
imageCalculator("Subtract create", idSigma1, idSigma2);


/////LAPLACIAN OF GAUSSIAN (MEXICAN HAT)
selectImage(idOrig);
run("FeatureJ Laplacian", "compute smoothing=2");
run("HiLo");
run("Invert LUT");


///MORPHOLOGICAL WHITE TOP HAT (STAN THAYER)
selectImage(idOrig);
run("Morphological Filters", "operation=[White Top Hat] element=Disk radius=5");

//The PSD95-GFP and Bassoon channels were filtered using a Laplacian of Gaussian with a sigma of 0.15 μM in X and Y, 0.45 μM in Z. A threshold of half the maximum intensity after filtering was used to identify regions of local maxima. Each connected component after thresholding was considered a separate punctum. For each punctum in the PSD95-GFP channel, we analyzed for the presence of a punctum in the bassoon channel by measuring the amount of overlap with the region above threshold. An overlap of 10% was considered apositive co-occurrence.
//https://www.frontiersin.org/articles/10.3389/fncel.2019.00467/full#F2
//DoGnet https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007012


///TOPHAT WITH FIND MAXIMA THRESHOLDING
idOrig = getImageID();
selectImage(idOrig);
run("FeatureJ Laplacian", "compute smoothing=2");
run("HiLo");
run("Invert LUT");
idLap = getImageID();


//LAP 1
selectImage(idLap);
run("Duplicate...", "title=[lap-maxima]");
run("Find Maxima...", "prominence=500 light output=[Single Points]");
idLapMax = getImageID();
//LAP 2
selectImage(idLap);
run("Duplicate...", "title=[lap-thresh]");
setAutoThreshold("Otsu");
setOption("BlackBackground", false);
run("Convert to Mask");
idLapThresh = getImageID();

run("Binary Feature Extractor", "objects=lap-thresh selector=[lap-maxima Maxima] object_overlap=1 combine count");