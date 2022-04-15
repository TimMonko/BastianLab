run("Set Measurements...", "area add redirect=None decimal=3");
dir = getDirectory("Choose a directory");
print(dir); 
list = getFileList(dir);
for (i=0; i<list.length; i++) { 
     if (endsWith(list[i], ".tif")){ 
               print(i + ": " + dir+list[i]); 
             open(dir+list[i]); 
             imgName=getTitle(); 
         baseNameEnd=indexOf(imgName, ".tif"); 
         baseName=substring(imgName, 0, baseNameEnd);
run("Properties...", "unit=µm pixel_width=.064 pixel_height=.064");
resetMinAndMax();
	//run("Z Project...", "start=5 stop=9 projection=[Max Intensity]");
	run("Z Project...", "projection=[Max Intensity]");
		saveAs("Tiff", dir+baseName + "_zprojmax.tif");
	run("Convolve...", "text1=[-3 -3 -3 -3 -3 -3 -3\n-3 -3 -3 2 -3 -3 -3\n-3 -3 2 15 2 -3 -3\n-3 2 15 23 15 2 -3\n-3 -3 2 15 2 -3 -3\n-3 -3 -3 2 -3 -3 -3\n-3 -3 -3 -3 -3 -3 -3]");
		saveAs("Tiff", dir+baseName + "_convolve.tif");
//run("Threshold...");
	//setAutoThreshold("Default dark");
	setAutoThreshold("Mean dark");
		run("Convert to Mask");
			saveAs("Tiff", dir+baseName + "_threshold.tif");
run("Analyze Particles...", "size=0.02-1.00 display clear summarize add");
	selectWindow("Results");
	saveAs("Results", dir+baseName + "_results.csv");
		close();
open(dir+list[i]);
	roiManager("Show All");
		saveAs("Tiff", dir+baseName + "_overlay.tif");
		close();
			run("Close All"); 
     } 
} 

selectWindow("Summary");
saveAs("Results", dir + "groupsummary.csv");