FromInput() ->
LoadBalanceByWeight("from-env") ->
IPsecESPencap() ->
IPsecHMACSHA1AES() -> 
L2Forward(method echoback) ->
ToOutput();
