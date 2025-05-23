import json
import requests
from requests.auth import HTTPBasicAuth

from grafana_api.grafana_face import GrafanaFace

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from prometheus_client.exposition import basic_auth_handler

GRAFANA_URL = '207.246.109.178:9091'
PROMETHUS_URL = '207.246.109.178:9001'
GRAFANA_API = '207.246.109.178:3005'
LOKI_API = '207.246.109.178:3100'
# docker run -d --name=loki --mount source=loki-data,target=/loki -p 3100:3100 grafana/loki
GRAFANA_USERNAME = 'admin'
GRAFANA_PASSWORD = 'admin123@'


class Promethus_Grafana:
    def __init__(self, host_grafana, url_grafana, url_promethus):
        self.username = GRAFANA_USERNAME
        self.password = GRAFANA_PASSWORD
        self.url_grafana = url_grafana
        self.url_promethus = url_promethus
        self.host_grafana = host_grafana

        self.grafana_api = GrafanaFace(
            auth=(self.username, self.password),
            host=url_grafana
        )

    def promethus_push_to(self, job_name):
        def my_auth_handler(url, method, timeout, headers, data):
            return basic_auth_handler(url, method, timeout, headers, data, self.username, self.password)

        registry = CollectorRegistry()
        g = Gauge('job_last_success_unixtime',
                  'Last time a batch job successfully finished', registry=registry)
        g.set_to_current_time()
        push_to_gateway(GRAFANA_URL, job=job_name,
                        registry=registry, handler=my_auth_handler)

    def create_dashboard(self, name_dashboard, title, job_query, tag):
        payload = {
            "dashboard": {
                "annotations": {
                    "list": [
                        {
                            "builtIn": 1,
                            "datasource": {
                                "type": "grafana",
                                "uid": "-- Grafana --"
                            },
                            "enable": True,
                            "hide": True,
                            "iconColor": "rgba(0, 211, 255, 1)",
                            "name": "Annotations & Alerts",
                            "type": "dashboard"
                        }
                    ]
                },
                "editable": True,
                "fiscalYearStartMonth": 0,
                "graphTooltip": 0,
                "id": 1,
                "links": [],
                "panels": [
                    {
                        "datasource": {
                            "type": "prometheus",
                            "uid": "aeet2lyrwxqf4e"
                        },
                        "fieldConfig": {
                            "defaults": {
                                "color": {
                                    "mode": "palette-classic"
                                },
                                "custom": {
                                    "axisBorderShow": False,
                                    "axisCenteredZero": False,
                                    "axisColorMode": "text",
                                    "axisLabel": "",
                                    "axisPlacement": "auto",
                                    "barAlignment": 0,
                                    "barWidthFactor": 0.6,
                                    "drawStyle": "line",
                                    "fillOpacity": 0,
                                    "gradientMode": "none",
                                    "hideFrom": {
                                        "legend": False,
                                        "tooltip": False,
                                        "viz": False
                                    },
                                    "insertNulls": False,
                                    "lineInterpolation": "linear",
                                    "lineWidth": 1,
                                    "pointSize": 5,
                                    "scaleDistribution": {
                                        "type": "linear"
                                    },
                                    "showPoints": "auto",
                                    "spanNulls": False,
                                    "stacking": {
                                        "group": "A",
                                        "mode": "none"
                                    },
                                    "thresholdsStyle": {
                                        "mode": "off"
                                    }
                                },
                                "mappings": [],
                                "thresholds": {
                                    "mode": "absolute",
                                    "steps": [
                                        {
                                            "color": "green",
                                            "value": None
                                        },
                                        {
                                            "color": "red",
                                            "value": 80
                                        }
                                    ]
                                }
                            },
                            "overrides": []
                        },
                        "gridPos": {
                            "h": 11,
                            "w": 12,
                            "x": 0,
                            "y": 0
                        },
                        "id": 4,
                        "options": {
                            "legend": {
                                "calcs": [],
                                "displayMode": "list",
                                "placement": "bottom",
                                "showLegend": True
                            },
                            "tooltip": {
                                "hideZeros": False,
                                "mode": "single",
                                "sort": "none"
                            }
                        },
                        "pluginVersion": "11.5.2",
                        "targets": [
                            {
                                "disableTextWrap": False,
                                "editorMode": "builder",
                                "expr": f'disk_used_gb{{job="{job_query}"}}',
                                "fullMetaSearch": False,
                                "includeNullMetadata": True,
                                "legendFormat": "__auto",
                                "range": True,
                                "refId": "A",
                                "useBackend": False
                            },
                            {
                                "datasource": {
                                    "type": "prometheus",
                                    "uid": "aeet2lyrwxqf4e"
                                },
                                "disableTextWrap": False,
                                "editorMode": "builder",
                                "expr": f'disk_used_gb{{job="{job_query}"}}',
                                "fullMetaSearch": False,
                                "hide": False,
                                "includeNullMetadata": True,
                                "instant": False,
                                "legendFormat": "__auto",
                                "range": True,
                                "refId": "B",
                                "useBackend": False
                            }
                        ],
                        "title": "Disk (Gb)",
                        "type": "timeseries"
                    },
                    {
                        "datasource": {
                            "type": "prometheus",
                            "uid": "aeet2lyrwxqf4e"
                        },
                        "fieldConfig": {
                            "defaults": {
                                "color": {
                                    "mode": "palette-classic"
                                },
                                "custom": {
                                    "axisBorderShow": False,
                                    "axisCenteredZero": False,
                                    "axisColorMode": "text",
                                    "axisLabel": "",
                                    "axisPlacement": "auto",
                                    "barAlignment": 0,
                                    "barWidthFactor": 0.6,
                                    "drawStyle": "line",
                                    "fillOpacity": 0,
                                    "gradientMode": "none",
                                    "hideFrom": {
                                        "legend": False,
                                        "tooltip": False,
                                        "viz": False
                                    },
                                    "insertNulls": False,
                                    "lineInterpolation": "linear",
                                    "lineWidth": 1,
                                    "pointSize": 5,
                                    "scaleDistribution": {
                                        "type": "linear"
                                    },
                                    "showPoints": "auto",
                                    "spanNulls": False,
                                    "stacking": {
                                        "group": "A",
                                        "mode": "none"
                                    },
                                    "thresholdsStyle": {
                                        "mode": "off"
                                    }
                                },
                                "mappings": [],
                                "thresholds": {
                                    "mode": "absolute",
                                    "steps": [
                                        {
                                            "color": "green",
                                            "value": None
                                        },
                                        {
                                            "color": "red",
                                            "value": 80
                                        }
                                    ]
                                }
                            },
                            "overrides": []
                        },
                        "gridPos": {
                            "h": 22,
                            "w": 12,
                            "x": 12,
                            "y": 0
                        },
                        "id": 3,
                        "options": {
                            "legend": {
                                "calcs": [],
                                "displayMode": "list",
                                "placement": "bottom",
                                "showLegend": True
                            },
                            "tooltip": {
                                "hideZeros": False,
                                "mode": "single",
                                "sort": "none"
                            }
                        },
                        "pluginVersion": "11.5.2",
                        "targets": [
                            {
                                "disableTextWrap": False,
                                "editorMode": "builder",
                                "expr": f'ram_total_gb{{job="{job_query}"}}',
                                "fullMetaSearch": False,
                                "includeNullMetadata": True,
                                "legendFormat": "__auto",
                                "range": True,
                                "refId": "A",
                                "useBackend": False
                            },
                            {
                                "datasource": {
                                    "type": "prometheus",
                                    "uid": "aeet2lyrwxqf4e"
                                },
                                "disableTextWrap": False,
                                "editorMode": "builder",
                                "expr": f'ram_used_gb{{job="{job_query}"}}',
                                "fullMetaSearch": False,
                                "hide": False,
                                "includeNullMetadata": True,
                                "instant": False,
                                "legendFormat": "__auto",
                                "range": True,
                                "refId": "B",
                                "useBackend": False
                            }
                        ],
                        "title": "RAM (Gb)",
                        "type": "timeseries"
                    },
                    {
                        "datasource": {
                            "type": "prometheus",
                            "uid": "aeet2lyrwxqf4e"
                        },
                        "fieldConfig": {
                            "defaults": {
                                "color": {
                                    "mode": "palette-classic"
                                },
                                "custom": {
                                    "axisBorderShow": False,
                                    "axisCenteredZero": False,
                                    "axisColorMode": "text",
                                    "axisLabel": "",
                                    "axisPlacement": "auto",
                                    "barAlignment": 0,
                                    "barWidthFactor": 0.6,
                                    "drawStyle": "line",
                                    "fillOpacity": 0,
                                    "gradientMode": "none",
                                    "hideFrom": {
                                        "legend": False,
                                        "tooltip": False,
                                        "viz": False
                                    },
                                    "insertNulls": False,
                                    "lineInterpolation": "linear",
                                    "lineWidth": 1,
                                    "pointSize": 5,
                                    "scaleDistribution": {
                                        "type": "linear"
                                    },
                                    "showPoints": "auto",
                                    "spanNulls": False,
                                    "stacking": {
                                        "group": "A",
                                        "mode": "none"
                                    },
                                    "thresholdsStyle": {
                                        "mode": "off"
                                    }
                                },
                                "mappings": [],
                                "thresholds": {
                                    "mode": "absolute",
                                    "steps": [
                                        {
                                            "color": "green",
                                            "value": None
                                        },
                                        {
                                            "color": "red",
                                            "value": 80
                                        }
                                    ]
                                }
                            },
                            "overrides": []
                        },
                        "gridPos": {
                            "h": 11,
                            "w": 12,
                            "x": 0,
                            "y": 11
                        },
                        "id": 2,
                        "options": {
                            "legend": {
                                "calcs": [],
                                "displayMode": "list",
                                "placement": "bottom",
                                "showLegend": True
                            },
                            "tooltip": {
                                "hideZeros": False,
                                "mode": "single",
                                "sort": "none"
                            }
                        },
                        "pluginVersion": "11.5.2",
                        "targets": [
                            {
                                "disableTextWrap": False,
                                "editorMode": "builder",
                                "expr": f'cpu_threads{{job="{job_query}"}}',
                                "fullMetaSearch": False,
                                "includeNullMetadata": True,
                                "legendFormat": "__auto",
                                "range": True,
                                "refId": "A",
                                "useBackend": False
                            },
                            {
                                "datasource": {
                                    "type": "prometheus",
                                    "uid": "aeet2lyrwxqf4e"
                                },
                                "disableTextWrap": False,
                                "editorMode": "builder",
                                "expr": f'cpu_cores{{job="{job_query}"}}',
                                "fullMetaSearch": False,
                                "hide": False,
                                "includeNullMetadata": True,
                                "instant": False,
                                "legendFormat": "__auto",
                                "range": True,
                                "refId": "B",
                                "useBackend": False
                            },
                            {
                                "datasource": {
                                    "type": "prometheus",
                                    "uid": "aeet2lyrwxqf4e"
                                },
                                "disableTextWrap": False,
                                "editorMode": "builder",
                                "expr": f'cpu_usage_percent{{job="{job_query}"}}',
                                "fullMetaSearch": False,
                                "hide": False,
                                "includeNullMetadata": True,
                                "instant": False,
                                "legendFormat": "__auto",
                                "range": True,
                                "refId": "C",
                                "useBackend": False
                            }
                        ],
                        "title": "CPU",
                        "type": "timeseries"
                    }
                ],
                "preload": False,
                "refresh": "auto",
                "schemaVersion": 40,
                "tags": [
                    "ml_llama3"
                ],
                "templating": {
                    "list": []
                },
                "time": {
                    "from": "now-30m",
                    "to": "now"
                },
                "timepicker": {},
                "timezone": "browser",
                "title": name_dashboard,
                "uid": "ceesj0cok1am8e",
                "version": 15,
                "weekStart": "",
                "tags": tag,
            },
            "overwrite": True
        }

        url = f"{self.host_grafana}/api/dashboards/db"

        # Headers
        headers = {
            'Accept-Language': 'en-US,en;q=0.9,vi;q=0.8',
            'accept': 'application/json, text/plain, */*'
        }

        # Thực hiện yêu cầu GET với Basic Auth
        res = requests.post(url, headers=headers, json=payload, auth=HTTPBasicAuth(
            GRAFANA_USERNAME, GRAFANA_PASSWORD))
        print(res)

    def generate_link_public(self, tag):
        dashboard_tag = self.grafana_api.search.search_dashboards(tag=tag)
        print(dashboard_tag)
        dashboard_uid = dashboard_tag[0]["uid"]
        print(dashboard_uid)

        # URL API để công khai Dashboard
        url = f"{self.host_grafana}/api/dashboards/uid/{dashboard_uid}/public-dashboards"

        # Headers
        headers = {
            'Accept-Language': 'en-US,en;q=0.9,vi;q=0.8',
            'accept': 'application/json, text/plain, */*'
        }

        # Thực hiện yêu cầu GET với Basic Auth
        requests.post(url, headers=headers, json={
                      "isEnabled": True}, auth=HTTPBasicAuth('admin', 'admin'))
        response = requests.get(url, headers=headers,
                                auth=HTTPBasicAuth('admin', 'admin'))

        if response.status_code == 200:
            res = response.json()
            link_url = f'http://{GRAFANA_API}/public-dashboards/{res["accessToken"]}'
        else:
            return False

        return link_url

# try:
#     promethus_grafana = Promethus_Grafana(
#         host_grafana=f"http://{GRAFANA_API}", url_grafana=GRAFANA_API, url_promethus=PROMETHUS_URL)

#     promethus_grafana.create_dashboard(
#         "ML_monitor_2", "Training Job", "flask-1", ["ml_m4t"])
#     print(promethus_grafana.generate_link_public('ml_m4t'))
# except Exception as e:
#     print(e)
