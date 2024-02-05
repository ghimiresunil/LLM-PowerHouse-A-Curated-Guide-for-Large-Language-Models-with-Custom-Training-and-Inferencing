from pymilvus import connections


class MilvusDBConnection(object):
    def __init__(self, alias="default"):
        host = "localhost"
        port = 19530
        self.alias = alias
        self.host = host
        self.port = port
        self.connetion = None

    def _build(self):
        self.connetion = connections.connect(
            alias=self.alias, host=self.host, port=self.port
        )

    def get_connections(self):
        if self.connetion is None:
            RuntimeError("Start the database connection!")
        return self.connetion

    def start(self):
        self._build()

    def stop(self):
        if self.connetion is not None:
            self.connetion.disconnect()
