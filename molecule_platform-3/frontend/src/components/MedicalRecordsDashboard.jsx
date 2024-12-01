import React, { useState } from 'react';
import {
  Box,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Text,
  Badge,
  VStack,
  Input,
  HStack,
  Select,
  Icon,
  Tooltip,
  Progress,
} from '@chakra-ui/react';
import { FiInfo, FiDatabase } from 'react-icons/fi';

const ResponseBadge = ({ response }) => {
  const colorScheme = {
    'Positive': 'green',
    'Negative': 'red',
    'Neutral': 'gray',
    'Mixed': 'yellow'
  }[response] || 'gray';

  return (
    <Badge
      colorScheme={colorScheme}
      px={2.5}
      py={0.5}
      borderRadius="full"
      fontSize="xs"
      textTransform="none"
      fontWeight="medium"
    >
      {response}
    </Badge>
  );
};

const MedicalRecordsDashboard = ({ records, isLoading }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterResponse, setFilterResponse] = useState('all');

  const filteredRecords = records?.filter(record => {
    const matchesSearch = record.diagnosis?.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         record.medication?.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesFilter = filterResponse === 'all' || record.discharge_disposition === filterResponse;
    return matchesSearch && matchesFilter;
  });

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
          <HStack spacing={3} mb={2}>
            <Icon as={FiDatabase} color="blue.400" boxSize={6} />
            <Text fontSize="xl" fontWeight="semibold" color="white">
              Medical Records Dashboard
            </Text>
          </HStack>
          <Box overflowX="auto">
            <Table variant="simple">
              <Tbody>
                <Tr>
                  <Td colSpan={5} textAlign="center" py={8}>
                    <VStack spacing={3}>
                      <Progress size="sm" isIndeterminate colorScheme="blue" w="200px" />
                      <Text color="gray.400">
                        Loading medical records...
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
        <HStack spacing={2} justify="space-between">
          <HStack>
            <Icon as={FiDatabase} color="blue.400" boxSize={5} />
            <Text fontSize="lg" fontWeight="semibold" color="white">
              Medical Records Dashboard
            </Text>
            <Tooltip label="Medical records related to similar molecular structures" hasArrow placement="top">
              <Box cursor="pointer">
                <Icon as={FiInfo} color="gray.400" boxSize={4} />
              </Box>
            </Tooltip>
          </HStack>
        </HStack>

        {records && (
          <HStack spacing={3}>
            <Input
              placeholder="Search by condition..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              size="sm"
              bg="whiteAlpha.50"
              borderColor="whiteAlpha.300"
              _hover={{ borderColor: 'whiteAlpha.400' }}
              _focus={{ borderColor: 'blue.400', boxShadow: '0 0 0 1px var(--chakra-colors-blue-400)' }}
              color="white"
              maxW="250px"
            />
            <Select
              value={filterResponse}
              onChange={(e) => setFilterResponse(e.target.value)}
              size="sm"
              bg="whiteAlpha.50"
              borderColor="whiteAlpha.300"
              _hover={{ borderColor: 'whiteAlpha.400' }}
              _focus={{ borderColor: 'blue.400', boxShadow: '0 0 0 1px var(--chakra-colors-blue-400)' }}
              color="white"
              maxW="150px"
              sx={{
                option: {
                  bg: 'navy.800',
                  color: 'white',
                  _hover: { bg: 'navy.700' }
                }
              }}
            >
              <option value="all" style={{color: 'black'}}>All Responses</option>
              <option value="Positive" style={{color: 'black'}}>Positive</option>
              <option value="Negative" style={{color: 'black'}}>Negative</option>
              <option value="Neutral" style={{color: 'black'}}>Neutral</option>
              <option value="Mixed" style={{color: 'black'}}>Mixed</option>
            </Select>
          </HStack>
        )}

        <Box
          flex="1"
          overflowX="auto"
          overflowY="auto"
          maxH="400px"
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
                {[
                  'ID',
                  'Patient Demographics',
                  'Diagnosis',
                  'Medication',
                  'Administration',
                  'Length of Stay',
                  'Clinical Outcome',
                  'Data Quality'
                ].map((header) => (
                  <Th
                    key={header}
                    color="gray.300"
                    borderColor="whiteAlpha.200"
                    fontSize="xs"
                    textTransform="uppercase"
                    letterSpacing="wider"
                    py={3}
                  >
                    {header}
                  </Th>
                ))}
              </Tr>
            </Thead>
            <Tbody>
              {records ? (
                filteredRecords.length > 0 ? (
                  filteredRecords.map((record) => (
                    <Tr key={record.hadm_id} _hover={{ bg: 'whiteAlpha.100' }}>
                      <Td borderColor="whiteAlpha.200">
                        <Text color="white" fontWeight="medium">{record.hadm_id}</Text>
                      </Td>
                      <Td borderColor="whiteAlpha.200">
                        <VStack align="start" spacing={1}>
                          <Text color="gray.300">{record.age_bucket}</Text>
                          <Text color="gray.400" fontSize="xs">
                            {record.gender} | {record.ethnicity}
                          </Text>
                        </VStack>
                      </Td>
                      <Td borderColor="whiteAlpha.200">
                        <VStack align="start" spacing={1}>
                          <Text color="gray.300">{record.diagnosis}</Text>
                          <Text color="blue.300" fontSize="xs">
                            ICD-9: {record.diagnosis_codes?.join(', ')}
                          </Text>
                        </VStack>
                      </Td>
                      <Td borderColor="whiteAlpha.200">
                        <VStack align="start" spacing={1}>
                          <Text color="gray.300">{record.medication}</Text>
                          <Text color="gray.400" fontSize="xs">
                            NDC: {record.ndc_code}
                          </Text>
                        </VStack>
                      </Td>
                      <Td borderColor="whiteAlpha.200">
                        <VStack align="start" spacing={1}>
                          <Text color="gray.300">{record.route}</Text>
                          <Text color="gray.400" fontSize="xs">
                            {record.dose_val_rx} {record.dose_unit_rx}
                          </Text>
                        </VStack>
                      </Td>
                      <Td borderColor="whiteAlpha.200">
                        <Text color="gray.300">{record.los} days</Text>
                      </Td>
                      <Td borderColor="whiteAlpha.200">
                        <Badge
                          colorScheme={
                            record.discharge_disposition === 'Home' ? 'green' :
                            record.discharge_disposition === 'Other Facility' ? 'yellow' :
                            record.discharge_disposition === 'Expired' ? 'red' : 'gray'
                          }
                          px={2}
                          py={0.5}
                          borderRadius="full"
                        >
                          {record.discharge_disposition}
                        </Badge>
                      </Td>
                      <Td borderColor="whiteAlpha.200">
                        <Text color="blue.300" fontSize="sm">
                          {record.data_quality}%
                        </Text>
                      </Td>
                    </Tr>
                  ))
                ) : (
                  <Tr>
                    <Td colSpan={8} textAlign="center" py={6}>
                      <Text color="gray.400" fontSize="sm">
                        No records match your search criteria
                      </Text>
                    </Td>
                  </Tr>
                )
              ) : (
                <Tr>
                  <Td colSpan={8} textAlign="center" py={6}>
                    <Text color="gray.400" fontSize="sm">
                      Analyze a molecule to view related medical records
                    </Text>
                  </Td>
                </Tr>
              )}
            </Tbody>
          </Table>
        </Box>
      </VStack>
    </Box>
  );
};

export default MedicalRecordsDashboard;
