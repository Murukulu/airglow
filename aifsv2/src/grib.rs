use std::{
    fs::File,
    io::{self, BufReader},
};

use grib::codetables::{CodeTable4_2, Lookup};

pub fn load_grib(path: &str) -> Result<(), io::Error> {
    let f = File::open(path)?;
    let g = grib::from_reader(BufReader::new(f))
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    for ((var_num, _), msg) in g.iter() {
        let _ = msg.describe();
        let discipline = msg.indicator().discipline;
        let category = msg.prod_def().parameter_category().unwrap();
        let num = msg.prod_def().parameter_number().unwrap();
        let name = CodeTable4_2::new(discipline, category).lookup(num as usize);
        println!("{}\t {:?}", var_num, name.to_string());
    }
    Ok(())
}
