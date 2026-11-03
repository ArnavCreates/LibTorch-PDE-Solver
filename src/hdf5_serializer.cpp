#include "pde_solver/hdf5_serializer.hpp"
#include <hdf5.h>
#include <iostream>
#include <filesystem>

namespace pde_solver {

bool HDF5Serializer::serialize_fields(
    const std::string& filepath,
    const ThermalFieldsPayload& payload,
    bool verbose
) {
    namespace fs = std::filesystem;
    fs::path p(filepath);
    if (p.has_parent_path()) {
        fs::create_directories(p.parent_path());
    }

    hid_t file_id = H5Fcreate(filepath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        std::cerr << "Error: Failed to create HDF5 file: " << filepath << std::endl;
        return false;
    }

    // Create groups /grid and /fields
    hid_t group_grid = H5Gcreate2(file_id, "/grid", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t group_fields = H5Gcreate2(file_id, "/fields", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    hsize_t nx = payload.x_coords.size();
    hsize_t ny = payload.y_coords.size();

    // 1. Write /grid/x and /grid/y
    hsize_t dim_x[1] = {nx};
    hid_t space_x = H5Screate_simple(1, dim_x, nullptr);
    hid_t dset_x = H5Dcreate2(group_grid, "x", H5T_IEEE_F64LE, space_x, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset_x, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, payload.x_coords.data());
    H5Dclose(dset_x);
    H5Sclose(space_x);

    hsize_t dim_y[1] = {ny};
    hid_t space_y = H5Screate_simple(1, dim_y, nullptr);
    hid_t dset_y = H5Dcreate2(group_grid, "y", H5T_IEEE_F64LE, space_y, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset_y, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, payload.y_coords.data());
    H5Dclose(dset_y);
    H5Sclose(space_y);

    // 2. Write 2D fields (nx, ny)
    hsize_t dims_2d[2] = {nx, ny};
    auto write_field = [&](hid_t grp, const char* name, const std::vector<double>& data) {
        hid_t space = H5Screate_simple(2, dims_2d, nullptr);
        hid_t dset = H5Dcreate2(grp, name, H5T_IEEE_F64LE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
        H5Dclose(dset);
        H5Sclose(space);
    };

    write_field(group_fields, "u_pinn", payload.u_pinn);
    write_field(group_fields, "u_fdm_gauss_seidel", payload.u_fdm_gauss_seidel);
    write_field(group_fields, "u_exact", payload.u_exact);
    write_field(group_fields, "abs_error_pinn_vs_fdm", payload.abs_error_pinn_vs_fdm);
    write_field(group_fields, "pde_residual_map", payload.pde_residual_map);

    // 3. Write Root Attributes
    hid_t scalar_space = H5Screate(H5S_SCALAR);
    auto write_attr = [&](const char* name, double val) {
        hid_t attr = H5Acreate2(file_id, name, H5T_IEEE_F64LE, scalar_space, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attr, H5T_NATIVE_DOUBLE, &val);
        H5Aclose(attr);
    };

    write_attr("alpha", payload.alpha);
    write_attr("final_time", payload.final_time);
    write_attr("relative_l2_error", payload.relative_l2_error);
    write_attr("boundary_rmse", payload.boundary_rmse);
    write_attr("energy_conservation_rel_error", payload.energy_conservation_rel_error);

    H5Sclose(scalar_space);
    H5Gclose(group_fields);
    H5Gclose(group_grid);
    H5Fclose(file_id);

    if (verbose) {
        std::cout << ">>> HDF5 Serialization Successful: " << filepath << " ["
                  << nx << " x " << ny << " tensors saved]\n";
    }

    return true;
}

} // namespace pde_solver
