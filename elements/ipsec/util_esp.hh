/*
 * esp.hh
 *
 *  Created on: Nov 20, 2011
 *      Author: seonggu
 */

#ifndef __NSHADER_UTIL_IPSEC_ESP_HH__
#define __NSHADER_UTIL_IPSEC_ESP_HH__

enum {
	// TODO: Shouldn't it be 16(= AES_BLOCK_SIZE)? why it was set to 8?
	ESP_IV_LENGTH = 16
};

struct esphdr {
	/* Security Parameters Index */
	uint32_t esp_spi;
	/* Replay counter */
	uint32_t esp_rpl;
	/* initial vector */
	uint8_t esp_iv[ESP_IV_LENGTH];
};

#endif /* __NSHADER_UTIL_IPSEC_ESP_HH_ */
