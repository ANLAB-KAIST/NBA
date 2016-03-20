import asyncio
import aiozmq, zmq

class LatencyHistogramReader:

    def __init__(self, loop=None):
        self._loop = loop if loop else asyncio.get_event_loop()
        self.records = []

    async def subscribe(self, remote_host, remote_cpu_idx):
        self.remote_addr = 'tcp://{}:{}'.format(remote_host, remote_cpu_idx)
        self._conn = await aiozmq.create_zmq_stream(zmq.SUB, loop=self._loop,
                                                    connect=self.remote_addr)
        self._conn.transport.setsockopt(zmq.SUBSCRIBE, b'')
        while True:
            try:
                recv_data = await self._conn.read()
            except asyncio.CancelledError:
                self._conn.close()
                break
            cpu_idx = int(recv_data[0].decode())
            elapsed_sec = int(recv_data[1].decode())
            # TODO: convert to sparse DataFrame
            histogram = recv_data[2].decode().splitlines()
            self.records.append((cpu_idx, elapsed_sec, histogram))

