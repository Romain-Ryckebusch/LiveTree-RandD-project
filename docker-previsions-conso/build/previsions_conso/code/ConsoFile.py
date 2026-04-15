# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 13:18:59 2020

@author: anthony.aouad
"""
import logging
from datetime import datetime, time, timedelta
# from time import perf_counter

import numpy as np
import pandas as pd
import pytz
import tensorflow as tf
from apscheduler.schedulers.blocking import BlockingScheduler
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
# x=loadmat('2019-2020_k.mat')
# x=x['Data_k'] #0)DateNum 1)Hôtel Académique | 2) HEI_13RT | 3) HEI_5RNS | 4) RIZOMM | 5) Ilôt 6) Temp
# x=np.delete(x,np.s_[1:5],axis=1)
from confluent_kafka import avro
from confluent_kafka.avro import AvroProducer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tzlocal import get_localzone

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s [%(name)-12s] %(levelname)-8s %(funcName)s: %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
class KafkaProducer(object):
    # Connection data
    _cls_bootstrap_servers_url = "kafka1:19092,kafka2:29092,kafka3:39092"
    _cls_schema_registry_url = "http://schema_registry:8081"
    #    _cls_security_protocol = 'ssl'
    #    _cls_ssl_ca_location = './ca-cert'
    #    _cls_ssl_certificate_location = './previsions_conso_livetree_signed.pem'
    #    _cls_ssl_key_location = './previsions_conso.key'
    # Application data
    # Data specific to series
    _cls_kafka_topics = ["CONSO_Prevision_Data"]
    _cls_series_topics = {
        "CONSO_Prevision_Data": _cls_kafka_topics[0],
    }
    _cls_series_keys = {
        "CONSO_Prevision_Data": "CONSO_Prevision_Data",
    }
    _cls_base_key_schema_str = """{
        "namespace": "my.test",
        "name": "key",
        "type": "record",
        "fields" : [
            {"name" : "serie", "type" : "string"}
        ]
    }
    """
    _cls_value_fields = {
        "CONSO_Prevision_Data": [
            {"name": "Ptot_HA_Forecast", "type": "float"},
            {"name": "Ptot_HEI_13RT_Forecast", "type": "float"},
            {"name": "Ptot_HEI_5RNS_Forecast", "type": "float"},
            {"name": "Ptot_Ilot_Forecast", "type": "float"},
            {"name": "Ptot_RIZOMM_Forecast", "type": "float"},
            {"name": "Date", "type": "string"},
        ]
    }
    _cls_value_schemas = [
        """
        {
        "namespace": "my.test",
        "name": "value",
        "type": "record",
        "fields" : [
            {"name": "Ptot_HA_Forecast", "type": ["null", "float"], "default": null},
            {"name": "Ptot_HEI_13RT_Forecast", "type": ["null", "float"], "default": null},
            {"name": "Ptot_HEI_5RNS_Forecast", "type": ["null", "float"], "default": null},
            {"name": "Ptot_Ilot_Forecast", "type": ["null", "float"], "default": null},
            {"name": "Ptot_RIZOMM_Forecast", "type": ["null", "float"], "default": null},
            {"name": "Date", "type": "string"}
        ]}"""
    ]
    _cls_value_schema_strings = {
        "CONSO_Prevision_Data": _cls_value_schemas[0],
    }

    def __init__(self, identification):
        # Initialise the variables
        self._identification = identification
        self._kafka_topic = KafkaProducer._cls_series_topics[identification]
        self._kafka_key = KafkaProducer._cls_series_keys[identification]
        self._key_schema_str = KafkaProducer._cls_base_key_schema_str
        self._value_schema_str = KafkaProducer._cls_value_schema_strings[identification]
        self._value_fields = KafkaProducer._cls_value_fields[identification]
        key_schema = avro.loads(self._key_schema_str)
        value_schema = avro.loads(self._value_schema_str)
        # Initialize the producer
        self.avroProducer = AvroProducer(
            {
                "bootstrap.servers": KafkaProducer._cls_bootstrap_servers_url,
                #            'security.protocol': KafkaProducer._cls_security_protocol,
                #            'ssl.ca.location': KafkaProducer._cls_ssl_ca_location,
                #            'ssl.certificate.location': KafkaProducer._cls_ssl_certificate_location,
                #            'ssl.key.location': KafkaProducer._cls_ssl_key_location,
                # 'ssl.key.password': KafkaProducer._cls_ssl_key_password,
                "on_delivery": self.delivery_report,
                "schema.registry.url": KafkaProducer._cls_schema_registry_url,
                #            'schema.registry.ssl.ca.location': KafkaProducer._cls_ssl_ca_location,
                #            'schema.registry.ssl.certificate.location': KafkaProducer._cls_ssl_certificate_location,
                #            'schema.registry.ssl.key.location': KafkaProducer._cls_ssl_key_location
            },
            default_key_schema=key_schema,
            default_value_schema=value_schema,
        )

    def produce_data(self, points_list):
        for point in points_list:
            key = {"serie": self._kafka_key}
            # point["Date"] = point["_row"]
            # value = {
            #     "TimeOfValue": point["_row"],
            #     "carbonIntensity": float(point["carbonIntensity"])
            # }
            value = dict()
            for field in self._value_fields:
                # Construct the entry for that point
                point_key = field["name"]
                point_value = point[field["name"]]
                # Cast the value for that point
                # if field["type"] == "string":   ## pas besoin
                #     point_value = str(point_value)
                # elif field["type"] == "float":
                #     point_value = float(point_value)
                # Put that point into the dictionary
                value[point_key] = point_value
            self.avroProducer.produce(topic=self._kafka_topic, value=value, key=key)
            self.avroProducer.flush()

    def delivery_report(self, err, msg):
        """Called once for each message produced to indicate delivery result.
        Triggered by poll() or flush()."""
        if err is not None:
            logger.error("Message delivery failed: {}".format(err))
        else:
            logger.info(
                "Message delivered to {} [{}]".format(msg.topic(), msg.partition())
            )


def TrainNNCons():
    def TrainParBat(df, bat, Nom):
        def utc_to_local(utc_dt):
            local_tz = get_localzone()
            local_dt = utc_dt.replace(tzinfo=pytz.utc).astimezone(local_tz)
            return local_tz.normalize(local_dt)  #

        def isweekend(xyz):
            if xyz.weekday() < 5:
                return 0
            else:
                return 1

        dfs = pd.read_excel("Holidays.xlsx", sheet_name=None)
        Entr = np.zeros(shape=((len(df) - 144 * 7), 15))
        for i in range(1008, len(df)):
            Entr[i - 1008, 0] = utc_to_local(df["Date"][i]).timetuple().tm_yday
            Entr[i - 1008, 1] = (
                utc_to_local(df["Date"][i]).minute
                + utc_to_local(df["Date"][i]).hour * 60
            )
            Entr[i - 1008, 2] = utc_to_local(df["Date"][i]).weekday()
            Entr[i - 1008, 3] = isweekend(utc_to_local(df["Date"][i]))
            Entr[i - 1008, 4] = (utc_to_local(df["Date"][i])).month
            if (
                dfs[str(utc_to_local(df["Date"][i]).year)]["Unnamed: 0"]
                == pd.Timestamp(utc_to_local(df["Date"][i]).strftime("%Y-%m-%d"))
            ).any():
                Entr[i - 1008, 5] = 1
            else:
                Entr[i - 1008, 5] = 0

            if (
                dfs[str(utc_to_local(df["Date"][i]).year)]["Unnamed: 2"]
                == pd.Timestamp(utc_to_local(df["Date"][i]).strftime("%Y-%m-%d"))
            ).any():
                Entr[i - 1008, 6] = 1
                Entr[i - 1008, 5] = 0
            else:
                Entr[i - 1008, 6] = 0
        Entr[:, 7] = df["Airtemp"][1008:]  ## AirTemp
        Entr[:, 8] = df[bat][:-1008]  ## j-7
        Entr[:, 9] = df[bat][864:-144]  ## j-1
        Entr[:, 10] = df[bat][720:-288]  ## j-2
        Entr[:, 11] = df[bat][576:-432]  ## j-3
        Entr[:, 12] = df[bat][432:-576]  ## j-4
        Entr[:, 13] = df[bat][288:-720]  ## j-5
        Entr[:, 14] = df[bat][144:-864]  ## j-6
        Tar = df[bat][1008:].to_numpy().reshape(-1, 1)

        ###### initiate scaler for normalising data
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        print(scaler_x.fit(Entr))
        xscale = scaler_x.transform(Entr)
        print(scaler_y.fit(Tar))
        yscale = scaler_y.transform(Tar)
        ###### split data into test and train randomized and 0.2 division
        X_train, X_test, y_train, y_test = train_test_split(
            xscale, yscale, test_size=0.25
        )
        ###### creating neural network base
        model = Sequential()
        model.add(
            Dense(1024, input_dim=15, kernel_initializer="normal", activation="relu")
        )
        model.add(Dense(512, activation="relu"))
        model.add(Dense(1, activation="linear"))
        model.summary()
        ######
        callback = tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=5, mode="min", restore_best_weights=True
        )
        model.compile(loss="mse", optimizer="adam", metrics=["mae"])
        history = model.fit(
            X_train,
            y_train,
            epochs=300,
            batch_size=720,
            callbacks=[callback],
            verbose=0,
            validation_data=(X_test, y_test),
        )
        ###### Print Loss and various predictions comparing to real and NN values
        print(history.history.keys())
        model.save(f"my_modelCons3{Nom}.h5")
        import joblib

        joblib.dump(scaler_y, f"scalerConso{Nom}.save")
        joblib.dump(scaler_x, f"scalerxConso{Nom}.save")

    # Cassandra get meteo data#
    ###################
    db_url = "10.64.253.10"
    username = "prev_so_mel"
    password = "xdf56@89"
    keyspace = "previsions_data"
    ###################
    auth_provider = PlainTextAuthProvider(username=username, password=password)
    cluster = Cluster([db_url], auth_provider=auth_provider)
    session = cluster.connect(keyspace)
    ###################
    def pandas_factory(colnames, rows):
        return pd.DataFrame(rows, columns=colnames)

    ###################
    session.row_factory = pandas_factory
    session.default_fetch_size = None
    ###################
    tod = datetime.utcnow() - timedelta(days=400)
    query = """SELECT * FROM conso_historiques_clean where name=%s and "Date" >= %s"""  #  x=(df1['Quality']=='0') Not needed for Conso.
    rslt = session.execute(query, ("Conso_Data", tod), timeout=None)
    df = rslt._current_rows
#pv_prev_meteo_clean modifier quand station m�teo fonctionnel par un historique modifier le 16/05/2025
    query = """SELECT "AirTemp" FROM pv_prev_meteo_clean where "Date" >= %s and "Date" <= %s ALLOW FILTERING"""
    rslt = session.execute(
        query,
        (
            df["Date"][0].strftime("%Y-%m-%dT%H:%M:%S%z"),
            df["Date"][len(df) - 1].strftime("%Y-%m-%dT%H:%M:%S%z"),
        ),
        timeout=None,
    )
    df1 = rslt._current_rows
    df["Airtemp"] = df1["AirTemp"].fillna(value=15)
    df["tot"] = (
        df["Ptot_HA"] + df["Ptot_HEI_13RT"] + df["Ptot_HEI_5RNS"] + df["Ptot_RIZOMM"]
    )
    # a=perf_counter()
    TrainParBat(df, "tot", "Tot")
    TrainParBat(df, "Ptot_HA", "HA_Puissances_Ptot")
    TrainParBat(df, "Ptot_HEI_13RT", "HEI_13RT_Puissances_Ptot")
    TrainParBat(df, "Ptot_HEI_5RNS", "HEI_5RNS_Puissances_Ptot")
    TrainParBat(df, "Ptot_RIZOMM", "Rizomm_TGBT_Puissances_Ptot")
    # print(a-perf_counter())


def MakePredConso():
    import joblib

    def PredParBat(Entr, df, bat, Nom):
        Entr[:, 8] = df[bat][:144]  ## j-7
        Entr[:, 9] = df[bat][864:]  ## j-1
        Entr[:, 10] = df[bat][720:864]  ## j-2
        Entr[:, 11] = df[bat][576:720]  ## j-3
        Entr[:, 12] = df[bat][432:576]  ## j-4
        Entr[:, 13] = df[bat][288:432]  ## j-5
        Entr[:, 14] = df[bat][144:288]  ## j-6
        model = tf.keras.models.load_model(f"my_modelCons{Nom}.h5")
        scaler_y = joblib.load(f"scalerConso{Nom}.save")
        scaler_x = joblib.load(f"scalerxConso{Nom}.save")
        xscale = scaler_x.transform(Entr)
        ytest1 = model.predict(xscale)
        Pmoy = scaler_y.inverse_transform(ytest1)
        return Pmoy

    def datetime_range(start, end, delta):
        current = start
        while current < end:
            yield current
            current += delta

    def isweekend(xyz):
        if xyz.weekday() < 5:
            return 0
        else:
            return 1

    ###### time range for tomorrow in local time
    # x = [
    #     dt
    #     for dt in datetime_range(
    #         pytz.timezone("Europe/Paris").localize(
    #             datetime.combine(datetime.today() + timedelta(days=1), time(0, 0))
    #         ),
    #         (
    #             pytz.timezone("Europe/Paris").localize(
    #                 datetime.combine(datetime.today() + timedelta(days=1), time(0, 0))
    #                 + timedelta(hours=23)
    #                 + timedelta(minutes=55)
    #             )
    #         ),
    #         timedelta(minutes=10),
    #     )
    # ]
    x = [
        dt
        for dt in datetime_range(
            pytz.timezone("Europe/Paris").localize(
                datetime.combine(datetime.today(), time(0, 0))
            ),
            (
                pytz.timezone("Europe/Paris").localize(
                    datetime.combine(datetime.today(), time(0, 0))
                    + timedelta(hours=23)
                    + timedelta(minutes=55)
                )
            ),
            timedelta(minutes=10),
        )
    ]
    if len(x) != 144:
        if len(x) == 150:
            x = x[6:]
        else:
            for i in range(0, 6):
                x.insert(24, x[23 - i])
    dfs = pd.read_excel("Holidays.xlsx", sheet_name=None)
    Entr = np.zeros(shape=((len(x)), 15))
    for i in range(0, len(Entr)):
        Entr[i, 0] = x[i].timetuple().tm_yday
        Entr[i, 1] = x[i].minute + x[i].hour * 60
        Entr[i, 2] = x[i].weekday()
        Entr[i, 3] = isweekend(x[i])
        Entr[i, 4] = (x[i]).month
        if (
            dfs[str(x[i].year)]["Unnamed: 0"] == pd.Timestamp(x[i].strftime("%Y-%m-%d"))
        ).any():
            Entr[i, 5] = 1
        else:
            Entr[i, 5] = 0

        if (
            dfs[str(x[i].year)]["Unnamed: 2"] == pd.Timestamp(x[i].strftime("%Y-%m-%d"))
        ).any():
            Entr[i, 6] = 1
            Entr[i, 5] = 0
        else:
            Entr[i, 6] = 0

    db_url = ["10.64.253.10", "10.64.253.11", "10.64.253.12"]
    username = "prev_so_mel"
    password = "xdf56@89"
    keyspace = "previsions_data"
    ###################
    auth_provider = PlainTextAuthProvider(username=username, password=password)
    cluster = Cluster(db_url, auth_provider=auth_provider)
    session = cluster.connect(keyspace)
    ###################
    def pandas_factory(colnames, rows):
        return pd.DataFrame(rows, columns=colnames)

    ###################
    session.row_factory = pandas_factory
    session.default_fetch_size = None
    ###################
    query = """SELECT "AirTemp" FROM pv_prev_meteo_clean where name=%s and "Date" >= %s and "Date" <= %s"""
    rslt = session.execute(
        query,
        (
            "Meteorological_Prevision_Data",
            x[0].astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%S%z"),
            x[-1].astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%S%z"),
        ),
        timeout=None,
    )
    df1 = rslt._current_rows
    Entr[:, 7] = df1["AirTemp"]
    query = """SELECT * FROM conso_historiques_clean where name=%s and "Date" >= %s and "Date" < %s"""
    ## daylight savings adjustment
    loc = pytz.timezone("Europe/Paris")
    reqtime = x[0].astimezone(pytz.utc) - timedelta(days=7)
    calctime = reqtime.astimezone(loc)
    if (loc.localize(datetime.combine(calctime, time())) - calctime).days < 0:
        finloc = reqtime - timedelta(hours=1)
    elif (loc.localize(datetime.combine(calctime, time())) - calctime).days > 0:
        finloc = reqtime + timedelta(hours=1)
    else:
        finloc = reqtime
    rslt = session.execute(
        query,
        (
            "Conso_Data",
            finloc.strftime("%Y-%m-%dT%H:%M:%S%z"),
            x[0].astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%S%z"),
        ),
        timeout=None,
    )
    df = rslt._current_rows
    if len(df) > 1008:
        df = df[-1008:]
    df["tot"] = (
        df["Ptot_HA"] + df["Ptot_HEI_13RT"] + df["Ptot_HEI_5RNS"] + df["Ptot_RIZOMM"]
    )
    if 850 < len(df) < 1008:
        df = (
            df[
                [
                    "Date",
                    "Ptot_HA",
                    "Ptot_HEI_13RT",
                    "Ptot_HEI_5RNS",
                    "Ptot_RIZOMM",
                    "tot",
                ]
            ]
            .set_index("Date")
            .resample("10Min")
            .asfreq()
            .interpolate()
            .reset_index()
        )
    dontot = PredParBat(Entr, df, "tot", "Tot")
    donha = PredParBat(Entr, df, "Ptot_HA", "HA_Puissances_Ptot")
    don13rt = PredParBat(Entr, df, "Ptot_HEI_13RT", "HEI_13RT_Puissances_Ptot")
    don5rns = PredParBat(Entr, df, "Ptot_HEI_5RNS", "HEI_5RNS_Puissances_Ptot")
    donrizo = PredParBat(Entr, df, "Ptot_RIZOMM", "Rizomm_TGBT_Puissances_Ptot")
    ProdList = []
    for i in range(0, len(x)):
        info = {
            "Ptot_HA_Forecast": int(donha[i]),
            "Ptot_HEI_13RT_Forecast": int(don13rt[i]),
            "Ptot_HEI_5RNS_Forecast": int(don5rns[i]),
            "Ptot_Ilot_Forecast": int(dontot[i]),
            "Ptot_RIZOMM_Forecast": int(donrizo[i]),
            "Date": x[i].astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        ProdList.append(info)
    KafkaProducer("CONSO_Prevision_Data").produce_data(ProdList)

#MakePredConso()
scheduler = BlockingScheduler()
scheduler.configure(timezone=pytz.timezone("Europe/Paris"))
scheduler.add_job(MakePredConso, "cron", hour="02", minute=10)
scheduler.add_job(TrainNNCons, "cron", day=15, hour=5)
scheduler.start()
