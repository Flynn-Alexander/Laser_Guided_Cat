% Assuming you have the cameraParams object from the Camera Calibrator

% Extract camera parameters
focalLength = cameraParams.FocalLength; % [fx, fy] in pixel units
principalPoint = cameraParams.PrincipalPoint; % [cx, cy] in pixel units
radialDistortion = cameraParams.RadialDistortion;
tangentialDistortion = cameraParams.TangentialDistortion;

% Open a file for writing
fileID = fopen('cameraParameters.txt', 'w');

% Check if the file was opened successfully
if fileID == -1
    error('File could not be opened');
end

% Write the parameters to the file
fprintf(fileID, 'Focal Length: fx = %.6f, fy = %.6f\n', focalLength(1), focalLength(2));
fprintf(fileID, 'Principal Point: cx = %.6f, cy = %.6f\n', principalPoint(1), principalPoint(2));
fprintf(fileID, 'Radial Distortion Coefficients: [%f, %f]\n', radialDistortion(1), radialDistortion(2));
fprintf(fileID, 'Tangential Distortion Coefficients: [%f, %f]\n', tangentialDistortion(1), tangentialDistortion(2));

% Close the file
fclose(fileID);

% Confirm file writing
disp('Camera parameters have been written to cameraParameters.txt');
