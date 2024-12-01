import React from 'react';
import {
  Box,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Text,
  VStack,
  Progress,
  Flex,
  Icon,
  Tooltip,
  HStack,
} from '@chakra-ui/react';
import { FiInfo, FiSearch } from 'react-icons/fi';

const SimilarityBar = ({ value }) => (
  <Flex align="center" w="100%">
    <Progress
      value={value * 100}
      size="sm"
      colorScheme="blue"
      borderRadius="full"
      flex="1"
      mr={2}
      bg="whiteAlpha.100"
      sx={{
        '& > div': {
          transition: 'all 0.3s'
        }
      }}
    />
    <Text fontSize="sm" color="blue.300" fontWeight="semibold" minW="45px">
      {(value * 100).toFixed(1)}%
    </Text>
  </Flex>
);

const SimilaritySearchResults = ({ results, isLoading }) => {
  if (isLoading) {
    return (
      <Box
        as="section"
        bg="whiteAlpha.50"
        borderRadius="xl"
        border="1px solid"
        borderColor="whiteAlpha.200"
        p={6}
      >
        <VStack spacing={4} align="stretch">
          <HStack spacing={2}>
            <Icon as={FiSearch} color="blue.400" boxSize={5} />
            <Text fontSize="lg" fontWeight="medium" color="white">
              Similarity Search Results
            </Text>
          </HStack>
          <Box overflowX="auto">
            <Table variant="simple">
              <Thead>
                <Tr>
                  <Th>Compound</Th>
                  <Th>Similarity</Th>
                  <Th>Activities</Th>
                  <Th>Source</Th>
                </Tr>
              </Thead>
              <Tbody>
                <Tr>
                  <Td colSpan={4} textAlign="center" py={8}>
                    <VStack spacing={3}>
                      <Progress size="sm" isIndeterminate colorScheme="blue" w="200px" />
                      <Text color="gray.400">
                        Searching for similar molecules...
                      </Text>
                    </VStack>
                  </Td>
                </Tr>
              </Tbody>
            </Table>
          </Box>
        </VStack>
      </Box>
    );
  }

  if (!results || !Array.isArray(results)) {
    return (
      <Box
        as="section"
        bg="whiteAlpha.50"
        borderRadius="xl"
        border="1px solid"
        borderColor="whiteAlpha.200"
        p={6}
        _hover={{ borderColor: 'whiteAlpha.300' }}
        transition="all 0.2s"
      >
        <VStack spacing={4} align="stretch">
          <HStack spacing={2}>
            <Icon as={FiSearch} color="blue.400" boxSize={5} />
            <Text fontSize="lg" fontWeight="medium" color="white">
              Similarity Search Results
            </Text>
          </HStack>
          <Box overflowX="auto">
            <Table variant="simple">
              <Thead>
                <Tr>
                  <Th>Compound</Th>
                  <Th>Similarity</Th>
                  <Th>Activities</Th>
                  <Th>Source</Th>
                </Tr>
              </Thead>
              <Tbody>
                <Tr>
                  <Td colSpan={4} textAlign="center" py={8}>
                    <VStack spacing={2}>
                      <Icon as={FiSearch} boxSize={6} color="blue.300" />
                      <Text color="gray.400">
                        Analyze a molecule to view similarity search results
                      </Text>
                    </VStack>
                  </Td>
                </Tr>
              </Tbody>
            </Table>
          </Box>
        </VStack>
      </Box>
    );
  }

  return (
    <Box
      as="section"
      bg="whiteAlpha.50"
      borderRadius="xl"
      border="1px solid"
      borderColor="whiteAlpha.200"
      p={6}
      _hover={{ borderColor: 'whiteAlpha.300' }}
      transition="all 0.2s"
      height="100%"
    >
      <VStack spacing={4} align="stretch" height="100%">
        <HStack spacing={3} justify="space-between">
          <HStack>
            <Icon as={FiSearch} color="blue.400" boxSize={5} />
            <Text fontSize="lg" fontWeight="semibold" color="white">
              Similarity Search Results
            </Text>
          </HStack>
          <Tooltip
            label="Compounds are ranked by structural similarity to the input molecule"
            hasArrow
            placement="top"
          >
            <Box cursor="pointer">
              <Icon as={FiInfo} color="gray.400" boxSize={4} />
            </Box>
          </Tooltip>
        </HStack>

        <Box
          flex="1"
          overflowX="auto"
          overflowY="auto"
          maxH="400px"
          bg="whiteAlpha.50"
          borderRadius="lg"
          css={{
            '&::-webkit-scrollbar': {
              width: '8px',
              height: '8px',
            },
            '&::-webkit-scrollbar-track': {
              background: 'rgba(255, 255, 255, 0.1)',
              borderRadius: '4px',
            },
            '&::-webkit-scrollbar-thumb': {
              background: 'rgba(255, 255, 255, 0.2)',
              borderRadius: '4px',
              '&:hover': {
                background: 'rgba(255, 255, 255, 0.3)',
              },
            },
          }}
        >
          <Table variant="simple" size="sm">
            <Thead position="sticky" top={0} bg="rgb(13, 18, 30)" zIndex={1}>
              <Tr>
                <Th
                  color="gray.300"
                  fontSize="xs"
                  textTransform="uppercase"
                  letterSpacing="wider"
                  pl={4}
                >
                  Compound
                </Th>
                <Th
                  color="gray.300"
                  fontSize="xs"
                  textTransform="uppercase"
                  letterSpacing="wider"
                >
                  Similarity
                </Th>
                <Th
                  color="gray.300"
                  fontSize="xs"
                  textTransform="uppercase"
                  letterSpacing="wider"
                >
                  Activities
                </Th>
                <Th
                  color="gray.300"
                  fontSize="xs"
                  textTransform="uppercase"
                  letterSpacing="wider"
                >
                  Source
                </Th>
              </Tr>
            </Thead>
            <Tbody>
              {results.map((result, index) => (
                <Tr key={index} _hover={{ bg: 'whiteAlpha.50' }}>
                  <Td pl={4}>
                    <VStack align="start" spacing={1}>
                      <Text color="white" fontWeight="medium">
                        {result.name || result.chembl_id || 'Unknown'}
                      </Text>
                      <Text fontSize="xs" color="blue.300" fontFamily="mono" opacity={0.8}>
                        {result.smiles}
                      </Text>
                      {result.properties && (
                        <Text fontSize="xs" color="gray.400">
                          MW: {result.properties.molecular_weight?.toFixed(1)} |
                          LogP: {result.properties.alogp?.toFixed(1)} |
                          PSA: {result.properties.psa?.toFixed(1)}
                        </Text>
                      )}
                    </VStack>
                  </Td>
                  <Td>
                    <SimilarityBar value={result.similarity} />
                    <Text fontSize="xs" color="gray.400" mt={1}>
                      Data Quality: {result.data_completeness?.toFixed(1)}%
                    </Text>
                  </Td>
                  <Td>
                    <VStack align="start" spacing={2}>
                      {result.medical_records?.usage_statistics && (
                        <Text fontSize="sm" color="gray.300">
                          • Clinical Usage: {result.medical_records.usage_statistics.total_prescriptions} cases
                        </Text>
                      )}
                      {result.side_effects?.effects_summary && (
                        <Text fontSize="sm" color="orange.300">
                          • Side Effects: {result.side_effects.effects_summary.total_effects} reported
                        </Text>
                      )}
                      {result.properties?.qed_weighted && (
                        <Text fontSize="sm" color="green.300">
                          • Drug-likeness Score: {result.properties.qed_weighted.toFixed(2)}
                        </Text>
                      )}
                    </VStack>
                  </Td>
                  <Td>
                    <VStack align="start" spacing={1}>
                      <Text color="gray.300">{result.source}</Text>
                      {result.references && (
                        <Text fontSize="xs" color="blue.300">
                          <a href={result.references.pubchem_url || result.references.chembl_url}
                             target="_blank"
                             rel="noopener noreferrer">
                            View in Database
                          </a>
                        </Text>
                      )}
                    </VStack>
                  </Td>
                </Tr>
              ))}
            </Tbody>
          </Table>
        </Box>
      </VStack>
    </Box>
  );
};

export default SimilaritySearchResults;
