use std::path::Path;

use super::*;

const METADATA_DIR: &str = "./data/quiet_grub/anemoi-metadata";

// The MARS block is what names every output field in GRIB, so pin one of each kind the writer
// distinguishes: a pressure level, a renamed surface field, and a wave-stream field whose block
// is the bare three-key form that inference.yaml's typed_variables patch produces.
#[test]
fn checkpoint_mars_identities() {
    let metadata = Metadata::load(Path::new(METADATA_DIR)).expect("checkpoint metadata");

    assert_eq!(
        metadata.mars["z_500"],
        Mars {
            param: "z".to_string(),
            levtype: "pl".to_string(),
            levelist: Some(500),
            stream: Some("oper".to_string()),
        }
    );
    assert_eq!(metadata.mars["snowc"].param, "fscov");
    assert_eq!(metadata.mars["snowc"].levelist, None);
    assert_eq!(metadata.mars["mwd"].stream.as_deref(), Some("wave"));

    // Every output channel must be encodable, cos/sin_mwd via their parent's block.
    for name in &metadata.output_channel_to_var {
        let name = name
            .strip_prefix("cos_")
            .or(name.strip_prefix("sin_"))
            .unwrap_or(name);
        assert!(metadata.mars.contains_key(name), "{name} has no mars block");
    }

    // The computed forcings are the only variables with no MARS identity.
    let without: Vec<_> = metadata
        .variables
        .iter()
        .filter(|name| !metadata.mars.contains_key(*name))
        .cloned()
        .collect();
    assert_eq!(without, metadata.computed_forcing);
}

#[test]
fn checkpoint_accumulations() {
    let metadata = Metadata::load(Path::new(METADATA_DIR)).expect("checkpoint metadata");
    assert_eq!(
        metadata.accumulations,
        ["cp", "ro", "sf", "ssrd", "strd", "tp"]
    );
}
