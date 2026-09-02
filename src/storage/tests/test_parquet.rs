use crate::storage::parquet::{
    load_dense_matrix, load_lambda, load_sparse_matrix, save_arrowspace_checkpoint_with_builder,
    save_dense_matrix, save_sparse_matrix,
};
use crate::storage::tests::test_storage::{
    create_test_builder, create_test_dense_matrix, create_test_dense_matrix_with_size,
    create_test_sparse_matrix, create_test_sparse_matrix_with_size,
};
use approx::assert_relative_eq;
use arrow::datatypes::SchemaRef;
use sprs::TriMat;
use tempfile::TempDir;

use arrow::{
    array::{Float64Array, RecordBatch, StringArray, UInt64Array},
    datatypes::{DataType, Field, Schema},
};

use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};

use smartcore::linalg::basic::{arrays::Array, matrix::DenseMatrix};

use std::{fs::File, path::Path, sync::Arc};

/// Helper function for *_multibatch tests
fn create_forced_multibatch_parquet(
    path: impl AsRef<Path>,
    schema: SchemaRef,
    batches: impl Iterator<Item = RecordBatch>,
) {
    let file = File::create(path).unwrap();
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(1024))
        .build();

    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();

    for batch in batches {
        writer.write(&batch).unwrap();
    }
    writer.close().unwrap();
}

#[test]
fn test_dense_roundtrip() {
    let temp_dir = TempDir::new().unwrap();

    let data = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![1e-3, 1e-5, 0.34235236234234],
    ];
    let original = DenseMatrix::from_2d_vec(&data).unwrap();

    save_dense_matrix(&original, temp_dir.path(), "test_dense", None).unwrap();

    let loaded = load_dense_matrix(temp_dir.path().join("test_dense.parquet")).unwrap();

    assert_eq!(original.shape(), loaded.shape());

    let (rows, cols) = original.shape();
    for i in 0..rows {
        for j in 0..cols {
            assert_relative_eq!(*original.get((i, j)), *loaded.get((i, j)), epsilon = 1e-10);
        }
    }
}

#[test]
fn test_sparse_roundtrip() {
    let temp_dir = TempDir::new().unwrap();

    let mut trimat = TriMat::new((4, 4));
    trimat.add_triplet(0, 0, 2.0);
    trimat.add_triplet(0, 1, -1.0);
    trimat.add_triplet(1, 1, 3.0);
    trimat.add_triplet(2, 2, 1.5);
    let original = trimat.to_csr();

    save_sparse_matrix(&original, temp_dir.path(), "test_sparse", None).unwrap();

    let loaded = load_sparse_matrix(temp_dir.path().join("test_sparse.parquet")).unwrap();

    assert_eq!(original.shape(), loaded.shape());
    assert_eq!(original.nnz(), loaded.nnz());

    for i in 0..4 {
        for j in 0..4 {
            let orig_val = original.get(i, j).copied().unwrap_or(0.0);
            let loaded_val = loaded.get(i, j).copied().unwrap_or(0.0);
            assert_relative_eq!(orig_val, loaded_val, epsilon = 1e-10);
        }
    }
}

#[test]
fn test_checkpoint_save_all_artifacts() {
    let temp_dir = TempDir::new().unwrap();
    let builder = create_test_builder();

    let raw_data = create_test_dense_matrix();
    let centroids = DenseMatrix::from_2d_vec(&vec![vec![1.5, 2.5, 3.5]]).unwrap();
    let adjacency = create_test_sparse_matrix();
    let laplacian = create_test_sparse_matrix();
    let signals = create_test_sparse_matrix();

    save_arrowspace_checkpoint_with_builder(
        temp_dir.path(),
        "checkpoint_test",
        &raw_data,
        &adjacency,
        &centroids,
        &laplacian,
        &signals,
        &builder,
    )
    .unwrap();

    // Verify all files exist
    let expected_files = vec![
        "checkpoint_test_raw_data.parquet",
        "checkpoint_test_adjacency.parquet",
        "checkpoint_test_centroids.parquet",
        "checkpoint_test_laplacian.parquet",
        "checkpoint_test_signals.parquet",
        "checkpoint_test_metadata.json",
    ];

    for filename in expected_files {
        let path = temp_dir.path().join(filename);
        assert!(path.exists(), "Missing file: {}", filename);
    }
}

#[test]
fn test_dense_multibatch() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path().join("test_dense_multibatch.parquet");

    // Create a matrix with 2000 rows. This exceeds the default Arrow batch size (1024),
    // forcing the reader to process multiple RecordBatches.
    let rows = 2000;
    let cols = 2;
    let matrix = create_test_dense_matrix_with_size(rows, cols);

    let schema = Arc::new(Schema::new(vec![
        Field::new("name_id", DataType::Utf8, false),
        Field::new("n_rows", DataType::UInt64, false),
        Field::new("n_cols", DataType::UInt64, false),
        Field::new("col_0", DataType::Float64, false),
        Field::new("col_1", DataType::Float64, false),
    ]));

    // Create batches
    let batches = (0..2).map(|i| {
        let start = i * 1000;
        let end = std::cmp::min(start + 1000, rows);
        let len = end - start;

        let name_arr = StringArray::from(vec!["test"; len]);
        let n_rows_arr = UInt64Array::from(vec![rows as u64; len]);
        let n_cols_arr = UInt64Array::from(vec![cols as u64; len]);

        let col0_data: Vec<f64> = (start..end).map(|r| matrix.get((r, 0)).clone()).collect();
        let col1_data: Vec<f64> = (start..end).map(|r| matrix.get((r, 1)).clone()).collect();

        let col0_arr = Float64Array::from(col0_data);
        let col1_arr = Float64Array::from(col1_data);

        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(name_arr),
                Arc::new(n_rows_arr),
                Arc::new(n_cols_arr),
                Arc::new(col0_arr),
                Arc::new(col1_arr),
            ],
        )
        .unwrap()
    });

    create_forced_multibatch_parquet(&path, schema.clone(), batches);

    // Verify that load_dense_matrix correctly concatenates all batches.
    let loaded = load_dense_matrix(path).unwrap();

    assert_eq!(loaded.shape(), (rows, cols), "Loaded matrix shape mismatch");
}

#[test]
fn test_sparse_multibatch() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path().join("test_sparse_multibatch.parquet");

    // Create a sparse matrix with 2000 rows.
    let rows = 2000;
    let cols = 10;
    let matrix = create_test_sparse_matrix_with_size(rows, cols);
    let nnz = matrix.nnz();

    let schema = Arc::new(Schema::new(vec![
        Field::new("name_id", DataType::Utf8, false),
        Field::new("n_rows", DataType::UInt64, false),
        Field::new("n_cols", DataType::UInt64, false),
        Field::new("nnz", DataType::UInt64, false),
        Field::new("row", DataType::UInt64, false),
        Field::new("col", DataType::UInt64, false),
        Field::new("value", DataType::Float64, false),
    ]));

    let chunk_size = 1000;
    let total_chunks = (nnz + chunk_size - 1) / chunk_size;

    let batches = (0..total_chunks).map(|i| {
        let start = i * chunk_size;
        let end = std::cmp::min(start + chunk_size, nnz);
        let len = end - start;

        let name_arr = StringArray::from(vec!["test"; len]);
        let n_rows_arr = UInt64Array::from(vec![rows as u64; len]);
        let n_cols_arr = UInt64Array::from(vec![cols as u64; len]);
        let nnz_arr = UInt64Array::from(vec![nnz as u64; len]);

        let mut chunk_rows = Vec::with_capacity(len);
        let mut chunk_cols = Vec::with_capacity(len);
        let mut chunk_vals = Vec::with_capacity(len);

        for idx in start..end {
            chunk_rows.push(idx as u64);
            chunk_cols.push((idx % cols) as u64);
            chunk_vals.push(1.0);
        }

        let row_arr = UInt64Array::from(chunk_rows);
        let col_arr = UInt64Array::from(chunk_cols);
        let val_arr = Float64Array::from(chunk_vals);

        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(name_arr),
                Arc::new(n_rows_arr),
                Arc::new(n_cols_arr),
                Arc::new(nnz_arr),
                Arc::new(row_arr),
                Arc::new(col_arr),
                Arc::new(val_arr),
            ],
        )
        .unwrap()
    });

    create_forced_multibatch_parquet(&path, schema.clone(), batches);

    let loaded = load_sparse_matrix(path).unwrap();

    // If bug exists, loaded.nnz() will be ~1024 instead of 2000
    assert_eq!(loaded.nnz(), nnz, "Loaded sparse matrix nnz mismatch");
    assert_eq!(
        loaded.shape(),
        (rows, cols),
        "Loaded sparse matrix shape mismatch"
    );
}

#[test]
fn test_lambda_multibatch() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path().join("test_lambda_multibatch.parquet");

    let n_values = 2000;
    let lambdas: Vec<_> = (0..n_values).map(|i| i as f64).collect();

    let schema = Arc::new(Schema::new(vec![
        Field::new("name_id", DataType::Utf8, false),
        Field::new("n_values", DataType::UInt64, false),
        Field::new("row_index", DataType::UInt64, false),
        Field::new("lambda", DataType::Float64, false),
    ]));

    let chunk_size = 1000;
    let total_chunks = (n_values + chunk_size - 1) / chunk_size;

    let batches = (0..total_chunks).map(|i| {
        let start = i * chunk_size;
        let end = std::cmp::min(start + chunk_size, n_values);
        let len = end - start;

        let name_arr = StringArray::from(vec!["test"; len]);
        let n_vals_arr = UInt64Array::from(vec![n_values as u64; len]);
        let row_idx_arr = UInt64Array::from((start as u64..end as u64).collect::<Vec<_>>());
        let lambda_arr = Float64Array::from(lambdas[start..end].to_vec());

        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(name_arr),
                Arc::new(n_vals_arr),
                Arc::new(row_idx_arr),
                Arc::new(lambda_arr),
            ],
        )
        .unwrap()
    });

    create_forced_multibatch_parquet(&path, schema.clone(), batches);

    let loaded = load_lambda(path).unwrap();

    assert_eq!(
        loaded.len(),
        n_values,
        "Loaded lambda vector length mismatch"
    );
}
