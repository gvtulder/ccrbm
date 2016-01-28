function extract_one(id)
% extract patches from one subject and save as a .mat file

fprintf('patient              '); disp(id);
[cells, labels, neighbourhoods] = extract_ild_patch([ 'ILD_DB_volumeROIs/', id ]);
patient_number_s = id;
if id(1) == 'H'
  patient_number_s = id(12:end);
end
fprintf('size(cells)          '); disp(size(cells));
fprintf('size(labels)         '); disp(size(labels));
fprintf('size(neighbourhoods) '); disp(size(neighbourhoods));
save([ 'ILD-cells/', patient_number_s, '.mat' ], 'cells', 'labels', 'neighbourhoods');

