// Google Earth Engine (Code Editor) script: CHIRPS daily rainfall at the 8 BRG stations.
//
// Run at https://code.earthengine.google.com, start both tasks in the Tasks tab, and
// save the two CSVs from Google Drive to data/external/ (see README).
//
// Bilinear sampling needs two things. resample('bilinear') alone is not enough: sampling
// at CHIRPS native scale puts each output pixel on the native grid and returns the cell
// value, i.e. nearest-cell. Sampling at SAMPLE_SCALE_M evaluates the surface at the point.

var RAIN_START = '2018-01-01'; // the 90-day window ending 2018-04-07 starts 2018-01-08
var RAIN_END_EXCLUSIVE = '2020-02-01'; // last day 2020-01-31
var CLIM_FIRST_YEAR = 1988;
var CLIM_LAST_YEAR = 2017; // the normal ends before the study period, so it cannot leak
var SAMPLE_SCALE_M = 100;

var stations = ee.FeatureCollection([
  ee.Feature(ee.Geometry.Point([103.88232, -1.7145]), {station: 'BRG3'}),
  ee.Feature(ee.Geometry.Point([103.91762, -1.7282]), {station: 'BRG4'}),
  ee.Feature(ee.Geometry.Point([104.02562, -1.3826]), {station: 'BRG5'}),
  ee.Feature(ee.Geometry.Point([103.9693, -1.44281]), {station: 'BRG6'}),
  ee.Feature(ee.Geometry.Point([109.47732, -0.195]), {station: 'BRG17'}),
  ee.Feature(ee.Geometry.Point([109.51923, -0.27979]), {station: 'BRG18'}),
  ee.Feature(ee.Geometry.Point([109.55062, -0.2985]), {station: 'BRG19'}),
  ee.Feature(ee.Geometry.Point([109.41272, -0.2402]), {station: 'BRG20'})
]);

var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY').select('precipitation');
var nativeProjection = ee.Image(chirps.first()).projection();

function sampleBilinear(image, outputName) {
  return image.resample('bilinear').reduceRegions({
    collection: stations,
    reducer: ee.Reducer.first().setOutputs([outputName]),
    scale: SAMPLE_SCALE_M
  });
}

// Daily series.
var daily = chirps.filterDate(RAIN_START, RAIN_END_EXCLUSIVE).map(function (image) {
  var date = image.date().format('YYYY-MM-dd');
  return sampleBilinear(image, 'precip_mm').map(function (feature) {
    return feature.set('date', date);
  });
}).flatten();

Export.table.toDrive({
  collection: daily,
  description: 'chirps_stations',
  fileNamePrefix: 'chirps_stations',
  fileFormat: 'CSV',
  selectors: ['station', 'date', 'precip_mm']
});

// Mean rainfall per day of year. A composite loses its projection, so restore it
// before resampling; otherwise bilinear would run on a default 1-degree grid.
var baseline = chirps.filter(ee.Filter.calendarRange(CLIM_FIRST_YEAR, CLIM_LAST_YEAR, 'year'));
var climatology = ee.FeatureCollection(ee.List.sequence(1, 366).map(function (doy) {
  var mean = baseline
    .filter(ee.Filter.calendarRange(doy, doy, 'day_of_year'))
    .mean()
    .setDefaultProjection(nativeProjection);
  return sampleBilinear(mean, 'precip_clim_mm').map(function (feature) {
    return feature.set('doy', doy);
  });
})).flatten();

Export.table.toDrive({
  collection: climatology,
  description: 'chirps_climatology',
  fileNamePrefix: 'chirps_climatology',
  fileFormat: 'CSV',
  selectors: ['station', 'doy', 'precip_clim_mm']
});

print('Stations', stations);
print('Daily images (expect 761)', chirps.filterDate(RAIN_START, RAIN_END_EXCLUSIVE).size());
