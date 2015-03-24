FromInput() ->
cl :: PacketSizeClassifier();

begin :: IPsecESPencap() ->
IPsecAES() ->
IPsecAuthHMACSHA1() -> 
L2Forward("method echoback") ->
ToOutput();

cl[0] -> GPUOnly() -> begin;
cl[1] -> LoadBalanceByWeight("from-env") -> begin;
cl[2] -> CPUOnly() -> begin;
