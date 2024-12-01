import React from 'react';
import { Box, Heading, useColorModeValue, Spinner, Center } from '@chakra-ui/react';

const AnalysisPanel = ({ title, children, isLoading = false }) => {
  const bgColor = useColorModeValue('rgba(13, 18, 30, 0.95)', 'rgba(13, 18, 30, 0.95)');
  const borderColor = useColorModeValue('whiteAlpha.200', 'whiteAlpha.200');

  return (
    <Box
      bg={bgColor}
      backdropFilter="blur(12px)"
      borderRadius="2xl"
      p={8}
      border="1px solid"
      borderColor={borderColor}
      boxShadow="0 8px 16px rgba(0, 0, 0, 0.2)"
      transition="all 0.3s ease-in-out"
      _hover={{
        borderColor: 'blue.400',
        transform: 'translateY(-2px)',
        boxShadow: '0 12px 24px rgba(0, 0, 0, 0.3)'
      }}
      mb={6}
      position="relative"
      minH="120px"
    >
      <Heading size="md" mb={6} color="blue.200" letterSpacing="wide">
        {title}
      </Heading>
      {isLoading ? (
        <Center h="100px">
          <Spinner size="lg" color="blue.400" thickness="3px" />
        </Center>
      ) : (
        children
      )}
    </Box>
  );
};

export default AnalysisPanel;
