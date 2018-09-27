using Microsoft.ML.Runtime.Api;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace DemandForBikeSharing
{
    public class BikeSharingDemandData
    {
        [ColumnName("Score")]
        public float PredictedCount;
    }

    public class BikeSharingDemandSample
    {
        [Column("2")] public float Season;
        [Column("3")] public float Year;
        [Column("4")] public float Month;
        [Column("5")] public float Hour;
        [Column("6")] public bool Holiday;
        [Column("7")] public float Weekday;
        [Column("8")] public float Weather;
        [Column("9")] public float Temperature;
        [Column("10")] public float NormalizedTemperature;
        [Column("11")] public float Humidity;
        [Column("12")] public float Windspeed;
        [Column("16")] public float Count;
    }

    public class BikeSharingDemandsCsvReader
    {
        public IEnumerable<BikeSharingDemandSample> GetDataFromCsv(string dataLocation)
        {
            return File.ReadAllLines(dataLocation)
                .Skip(1)
                .Select(x => x.Split(','))
                .Select(x => new BikeSharingDemandSample()
                {
                    Season = float.Parse(x[2]),
                    Year = float.Parse(x[3]),
                    Month = float.Parse(x[4]),
                    Hour = float.Parse(x[5]),
                    Holiday = int.Parse(x[6]) != 0,
                    Weekday = float.Parse(x[7]),
                    Weather = float.Parse(x[8]),
                    Temperature = float.Parse(x[9]),
                    NormalizedTemperature = float.Parse(x[10]),
                    Humidity = float.Parse(x[11]),
                    Windspeed = float.Parse(x[12]),
                    Count = float.Parse(x[15])
                });
        }
    }
}
