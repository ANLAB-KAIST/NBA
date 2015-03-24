FromInput() -> c :: PacketSizeClassifier();
c[0] -> L2Forward(method echoback) -> ToOutput();
c[1] -> Discard();
c[2] -> L2Forward(method roundrobin_batch) -> ToOutput();
