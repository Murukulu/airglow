use std::{error, f64::consts::PI, fmt};

use burn::prelude::*;
use chrono::{DateTime, Datelike, NaiveTime, TimeZone, Timelike};

#[derive(Debug)]
pub enum Error {
    DateTime(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::DateTime(reason) => write!(f, "datetime: {}", reason),
        }
    }
}

impl error::Error for Error {}

fn deg2rad<B: Backend>(x: Tensor<B, 1>) -> Tensor<B, 1> {
    (x * PI) / 180.
}

// Fractional day of year, zero-based: Jan 1 00:00 is 0.0, Jan 2 12:00 is 1.5. Not the
// astronomical Julian Day Number, despite the name earthkit gives it.
fn julian_day<Tz: TimeZone>(date: &DateTime<Tz>) -> Result<f64, Error> {
    // Convert to utc and get start of the year. We find the difference here to
    // compute the number of days (+fractional in seconds) since start of the year.
    //
    // with_ordinal0 keeps the time of day, so it alone would land on Jan 1 at the *current*
    // hour and the fractional part -- the whole point of this function -- would always be
    // zero. Dropping to the NaiveDate and back through and_time is what forces midnight.
    let date = date.naive_utc();
    let year_start = date
        .date()
        .with_ordinal0(0)
        .ok_or(Error::DateTime("unable to get start of year"))?
        .and_time(NaiveTime::MIN);
    Ok((date - year_start).num_seconds() as f64 / 86400.)
}

fn local_time<B: Backend, Tz: TimeZone>(long: Tensor<B, 1>, date: &DateTime<Tz>) -> Tensor<B, 1> {
    let date = date.naive_utc();
    let day_start = date.date().and_time(NaiveTime::MIN);

    let hours_since_midnight = (date - day_start).num_seconds() as f64 / 3600.;
    // TODO(saiputravu): Maybe we shouldn't use modulo here?
    // Actually I should look at how fmod is implemented.
    ((long / 360. * 24.) + hours_since_midnight) % 24
}

// This computes the cos(solar zenith angle). The maths is slightly beyond me
// and it doesn't help that earthkit does not document the code.
//
// The solar zenith angle is the angle between the sun and the various point.
fn insolation<B: Backend, Tz: TimeZone>(
    lat: Tensor<B, 1>,
    long: Tensor<B, 1>,
    date: &DateTime<Tz>,
) -> Result<Tensor<B, 1>, Error> {
    assert_eq!(
        lat.shape(),
        long.shape(),
        "latitudes and longitudes shapes must match"
    );

    let angle = julian_day(date)? / DAYS_IN_A_YEAR * 2. * PI;

    // I have literally no idea where these numbers come from.
    // TODO(saiputravu): Figure out why earthkit-meteo uses these values in
    // their calculation.

    // Declination in degrees.
    let declination = 0.396372 - 22.91327 * f64::cos(angle) + 4.025430 * f64::sin(angle)
        - 0.387205 * f64::cos(2. * angle)
        + 0.051967 * f64::sin(2. * angle)
        - 0.154527 * f64::cos(3. * angle)
        + 0.084798 * f64::sin(3. * angle);

    // time correction in [ h.degrees ]
    let time_correction = 0.004297 + 0.107029 * f64::cos(angle)
        - 1.837877 * f64::sin(angle)
        - 0.837378 * f64::cos(2. * angle)
        - 2.340475 * f64::sin(2. * angle);

    // Both are scalars, so the trig stays on the host: one f64 sin, not a sin kernel over every
    // node in the grid. Only the per-node lat/long terms need to be tensors.
    let lat = deg2rad(lat);
    let sindec_sinlat = declination.to_radians().sin() * lat.clone().sin();
    let cosdec_coslat = declination.to_radians().cos() * lat.cos();

    // Compute the solar hour angle. `long` stays in degrees here -- unlike `lat` above -- because
    // the hour angle is built in degrees and converted once at the end. earthkit reads whole
    // hours only, so minutes are discarded; that is faithful, not an oversight.
    let hour = date.naive_utc().hour() as f64;
    let solar_angle = deg2rad((hour - 12.) * 15. + long + time_correction);
    let zenith_angle = sindec_sinlat + cosdec_coslat * solar_angle.cos();

    // Remove negative values. I guess this is for nodes with nighttime.
    Ok(zenith_angle.clamp_min(0.))
}

const DAYS_IN_A_YEAR: f64 = 365.25;

// Names match the earthkit variable names, so these can be looked up against
// `Metadata::variables` without a translation table.
pub struct Forcings<B: Backend> {
    pub sin_latitude: Tensor<B, 1>,
    pub cos_latitude: Tensor<B, 1>,
    pub sin_longitude: Tensor<B, 1>,
    pub cos_longitude: Tensor<B, 1>,
    pub sin_julian_day: Tensor<B, 1>,
    pub cos_julian_day: Tensor<B, 1>,
    pub sin_local_time: Tensor<B, 1>,
    pub cos_local_time: Tensor<B, 1>,
    pub insolation: Tensor<B, 1>,
}

// lat -> [N] of all points in the grid, degrees
// long -> [N] of all points in the grid, degrees
//
// Nine [N] fields, one per computed forcing.
pub fn compute_forcings<B: Backend, Tz: TimeZone>(
    lat: Tensor<B, 1>,
    long: Tensor<B, 1>,
    date: &DateTime<Tz>,
) -> Result<Forcings<B>, Error> {
    assert_eq!(
        lat.shape(),
        long.shape(),
        "latitudes and longitudes shapes must match"
    );

    // Lat/long forcings (4).
    let sin_latitude = deg2rad(lat.clone()).sin();
    let cos_latitude = deg2rad(lat.clone()).cos();
    let sin_longitude = deg2rad(long.clone()).sin();
    let cos_longitude = deg2rad(long.clone()).cos();

    // Julian day forcings (2). Constant across the grid, so the trig is one host f64 call and
    // the result is broadcast, rather than two kernels over every node.
    let device = lat.device();
    let julian_day_radians = julian_day(date)? / DAYS_IN_A_YEAR * 2. * PI;
    let sin_julian_day = Tensor::<B, 1>::full(lat.shape(), julian_day_radians.sin(), &device);
    let cos_julian_day = Tensor::<B, 1>::full(lat.shape(), julian_day_radians.cos(), &device);

    // Local time forcings (2).
    let local_time_radians = local_time(long.clone(), date) / 24 * 2 * PI;
    let sin_local_time = local_time_radians.clone().sin();
    let cos_local_time = local_time_radians.cos();

    // Solar zenith forcings (1).
    let insolation = insolation(lat, long, date)?;

    Ok(Forcings {
        sin_latitude,
        cos_latitude,
        sin_longitude,
        cos_longitude,
        sin_julian_day,
        cos_julian_day,
        sin_local_time,
        cos_local_time,
        insolation,
    })
}

#[cfg(test)]
#[path = "forcings_test.rs"]
mod tests;
