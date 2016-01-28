%
% ILD patch extraction similar to
% Li, Q., Cai, W., & Feng, D. D. (2013). Lung Image Patch Classification with Automatic Feature Learning. In The 35th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC 2013).
%
% for each slice,
%  - split in 32x32-pixel patches with 16px overlap
%  - add to set if >75% belongs to the same class
%

function [patches,labels,neighbourhoods] = extract_ild_patch(dirname)

roi_dcms = dir([ dirname, '/roi_mask/*.dcm' ]);
roi_slice_locations = [];
for i = 1:length(roi_dcms)
  Y_info = dicominfo([ dirname, '/roi_mask/', roi_dcms(i).name ]);
  roi_slice_locations = [ roi_slice_locations ; Y_info.SliceLocation ];
end

patches = [];
labels = [];
neighbourhoods = [];

slice_dcms = dir([ dirname, '/*.dcm' ]);
for i = 1:length(slice_dcms)
  slice_dcm = [ dirname, '/', slice_dcms(i).name ];
  X_info = dicominfo(slice_dcm);
  roi_idx = find(X_info.SliceLocation == roi_slice_locations);
  if length(roi_idx) ~= 1
    X_info.SliceLocation
    roi_slice_locations
    error('SliceLocation not found.')
  end
  roi_dcm = [ dirname, '/roi_mask/', roi_dcms(roi_idx).name ];

  [ p, l, n ] = load_patches(slice_dcm, roi_dcm);
  patches = [ patches, p ];
  labels  = [ labels, l ];
  neighbourhoods = [ neighbourhoods, n ];
end

end


function [patches, labels, neighbourhoods] = load_patches(slice_dcm, roi_dcm)

X = dicomread(slice_dcm);
X_info = dicominfo(slice_dcm);
Y = dicomread(roi_dcm);
Y_info = dicominfo(roi_dcm);

% fprintf('Rescale: slope = %f   intercept = %f\n', X_info.RescaleSlope, X_info.RescaleIntercept);

X_double = double(X) * X_info.RescaleSlope + X_info.RescaleIntercept;

if X_info.SliceLocation ~= Y_info.SliceLocation
  error('Slice location does not match.');
end

patches = [];
labels = [];
neighbourhoods = [];

for x_offset = 1:16:(512-32)
  for y_offset = 1:16:(512-32)
    X_patch = X_double(y_offset:(y_offset+31), x_offset:(x_offset+31));
    Y_patch = Y(y_offset:(y_offset+31), x_offset:(x_offset+31));
    label = unique(Y_patch(:));
    label = label(label ~= 0);
    if length(label) == 1
      prop = sum(Y_patch(:) == label) / prod(size(Y_patch));
      if prop >= 0.75
        cell_size = size(X_patch);
        neighbourhood = zeros(3*cell_size(1), 3*cell_size(2));

        from_y = y_offset - cell_size(1); to_y = y_offset + 2 * cell_size(1) - 1;
        from_y_r = max(from_y, 1); to_y_r = min(to_y, size(X_double, 1));
        from_x = x_offset - cell_size(2); to_x = x_offset + 2 * cell_size(2) - 1;
        from_x_r = max(from_x, 1); to_x_r = min(to_x, size(X_double, 2));

        neigh_y = (from_y_r - from_y + 1);
        neigh_x = (from_x_r - from_x + 1);

        neighbourhood(neigh_y:(neigh_y + (to_y_r - from_y_r)), ...
                      neigh_x:(neigh_x + (to_x_r - from_x_r))) ...
               = X_double(from_y_r:to_y_r, from_x_r:to_x_r);

        patches = [ patches, X_patch(:) ];
        labels  = [ labels, label ];
        neighbourhoods = [ neighbourhoods, neighbourhood(:) ];
      end
    end
  end
end

end

