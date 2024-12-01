import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  VStack,
  Input,
  Button,
  Text,
  Flex,
  Icon,
  Heading,
  Divider,
} from '@chakra-ui/react';
import { FiSend, FiUser, FiCpu, FiMessageCircle } from 'react-icons/fi';

const Message = ({ content, isUser }) => (
  <Flex justify={isUser ? 'flex-end' : 'flex-start'} w="100%" mb={4}>
    <Flex
      maxW="80%"
      bg={isUser ? 'blue.600' : 'whiteAlpha.200'}
      color={isUser ? 'white' : 'gray.100'}
      p={5}
      borderRadius="2xl"
      alignItems="flex-start"
      boxShadow="lg"
    >
      <Icon
        as={isUser ? FiUser : FiCpu}
        mr={3}
        mt={1}
        color={isUser ? 'white' : 'blue.300'}
        boxSize={5}
      />
      <Text fontSize="md" lineHeight="1.6">{content}</Text>
    </Flex>
  </Flex>
);

const SynthMolAssistant = ({ moleculeData, isLoading: parentLoading }) => {
  const [messages, setMessages] = useState([
    {
      content: "I'm ready to help analyze molecular structures and answer questions about the analysis results. What would you like to know?",
      isUser: false
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (moleculeData && !parentLoading) {
      const analysis = generateInitialAnalysis(moleculeData);
      setMessages(prev => [...prev, {
        content: analysis,
        isUser: false
      }]);
    }
  }, [moleculeData, parentLoading]);

  const generateInitialAnalysis = (data) => {
    const {
      similarity_search_results,
      risk_assessment,
      side_effects,
      medical_records
    } = data;

    const similarCompounds = similarity_search_results?.length || 0;
    const topSimilarity = similarity_search_results?.[0]?.similarity || 0;
    const riskLevel = risk_assessment?.toxicity_score ?
      `${(risk_assessment.toxicity_score * 100).toFixed(1)}%` : 'unknown';
    const sideEffectsCount = side_effects?.length || 0;
    const clinicalCases = medical_records?.length || 0;

    return `I've analyzed the molecule and found:
• ${similarCompounds} similar compounds in databases (top similarity: ${(topSimilarity * 100).toFixed(1)}%)
• Overall toxicity risk level: ${riskLevel}
• ${sideEffectsCount} potential side effects identified
• ${clinicalCases} relevant clinical cases found

You can ask me specific questions about:
- Similar compounds and their properties
- Risk assessment details and molecular properties
- Potential side effects and their severity
- Clinical usage patterns and outcomes
- Structure-activity relationships`;
  };

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput('');
    setMessages(prev => [...prev, { content: userMessage, isUser: true }]);
    setIsLoading(true);

    try {
      // Format the molecule data to include all analysis results
      const contextData = {
        molecule_data: {
          ...moleculeData,
          similarity_search: {
            similar_compounds: moleculeData.similarity_search_results?.map(result => ({
              name: result.name,
              similarity: result.similarity,
              properties: result.properties,
              activities: result.activities
            }))
          },
          risk_metrics: {
            toxicity_score: moleculeData.risk_assessment?.toxicity_score,
            bioavailability: moleculeData.risk_assessment?.bioavailability_score,
            reactivity: moleculeData.risk_assessment?.reactivity_score,
            environmental_impact: moleculeData.risk_assessment?.environmental_score
          },
          side_effects: moleculeData.side_effects?.map(effect => ({
            name: effect.name,
            severity: effect.severity,
            frequency: effect.frequency,
            evidence_level: effect.evidence_level
          })),
          clinical_data: moleculeData.medical_records?.map(record => ({
            condition: record.condition,
            treatment: record.treatment,
            outcome: record.outcome,
            duration: record.duration,
            confidence: record.confidence
          }))
        }
      };

      const response = await fetch('https://api.aimlapi.com/api/v1/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer 4c52a9817b884ef18597d098474086c1'
        },
        body: JSON.stringify({
          question: userMessage,
          context: contextData
        }),
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }

      const data = await response.json();
      setMessages(prev => [...prev, { content: data.response, isUser: false }]);
    } catch (error) {
      console.error('Chat error:', error);
      setMessages(prev => [...prev, {
        content: "I apologize, but I encountered an error processing your request. Please try again or rephrase your question.",
        isUser: false
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <VStack h="full" spacing={0} bg="gray.800" borderRadius="xl" borderWidth="1px" borderColor="whiteAlpha.200">
      <Flex w="full" p={5} borderBottomWidth="1px" borderColor="whiteAlpha.200" bg="whiteAlpha.50">
        <Icon as={FiMessageCircle} color="blue.400" boxSize={6} mr={3} />
        <Heading size="md" color="white">SynthMol Assistant</Heading>
      </Flex>

      <Box
        flex="1"
        w="full"
        overflowY="auto"
        p={6}
        css={{
          '&::-webkit-scrollbar': {
            width: '6px',
          },
          '&::-webkit-scrollbar-track': {
            width: '8px',
            background: 'rgba(255, 255, 255, 0.05)',
          },
          '&::-webkit-scrollbar-thumb': {
            background: 'rgba(255, 255, 255, 0.2)',
            borderRadius: '24px',
          },
        }}
      >
        <VStack spacing={4} align="stretch">
          {messages.map((message, index) => (
            <Message key={index} {...message} />
          ))}
          <div ref={messagesEndRef} />
        </VStack>
      </Box>

      <Divider borderColor="whiteAlpha.200" />

      <Flex w="full" p={5} bg="whiteAlpha.50">
        <Input
          placeholder="Ask about molecular analysis results..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          bg="gray.700"
          borderColor="whiteAlpha.300"
          _hover={{ borderColor: 'whiteAlpha.400' }}
          _focus={{ borderColor: 'blue.400', boxShadow: '0 0 0 1px var(--chakra-colors-blue-400)' }}
          color="white"
          mr={3}
          size="lg"
          fontSize="md"
        />
        <Button
          colorScheme="blue"
          onClick={handleSend}
          isLoading={isLoading}
          leftIcon={<Icon as={FiSend} boxSize={5} />}
          px={8}
          size="lg"
        >
          Send
        </Button>
      </Flex>
    </VStack>
  );
};

export default SynthMolAssistant;
