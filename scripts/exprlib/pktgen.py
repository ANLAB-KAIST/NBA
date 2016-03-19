import asyncio
import aiozmq, zmq
import simplejson as json

class PktGenController:

    def __init__(self, loop=None):
        self._loop = loop if loop else asyncio.get_event_loop()
        self._conn = None
        self._args = []

    async def init(self):
        self._conn = await aiozmq.create_zmq_stream(zmq.PUB, loop=self._loop,
                                                    bind='tcp://*:54321')
        # We need some time to wait until pub/sub endpoints are connected.
        await asyncio.sleep(0.5)

    @property
    def args(self):
        return self._args

    @args.setter
    def args(self, value):
        assert isinstance(value, tuple) or isinstance(value, list)
        self._args = value

    async def start(self, read_latencies=False):
        self._conn.write([json.dumps({
            'action': 'START',
            'args': self._args,
            'read_latencies': read_latencies,
        }).encode()])
        await self._conn.drain()

    async def stop(self):
        self._conn.write([json.dumps({
            'action': 'STOP',
        }).encode()])
        await self._conn.drain()

    async def __aenter__(self):
        await self.start()

    async def __aexit__(self, exc_type, exc, tb):
        await self.stop()
